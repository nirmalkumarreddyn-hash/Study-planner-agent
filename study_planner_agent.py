# # AI Study Planner & Performance Agent
# A three-agent ReAct system (Diagnostic, Scheduler, Evaluator) coordinated by a state-machine orchestrator.
#
# **Run it:** Runtime > Run all. The last cell prints a local link and a public `gradio.live` link.
#
# **API key (optional):** put `OPENAI_API_KEY` in Colab Secrets (key icon, left sidebar) or paste it in cell 2.
# With no key the app runs on a built-in rule-based engine, so nothing crashes.

# %%
# Cell 1: install dependencies
#!pip install -q "gradio>=5,<6" openai pydantic
# %%
# Cell 2: optional OpenAI key (leave empty to use the offline mock engine)
import os

OPENAI_API_KEY = ""   # e.g. "sk-..."  (or add a Colab Secret named OPENAI_API_KEY)

if OPENAI_API_KEY.strip():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.strip()
else:
    try:
        from google.colab import userdata  # type: ignore
        os.environ.setdefault("OPENAI_API_KEY", userdata.get("OPENAI_API_KEY"))
    except Exception:
        pass  # no secret configured: fine, mock engine will be used

# %%
# Cell 3: schemas, knowledge graph, LLM wrapper
import copy, hashlib, html, json, math
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
from pydantic import BaseModel, Field

SESSION_MIN, BREAK_MIN, LONG_BREAK_MIN = 45, 10, 20
INTERVALS = [1, 3, 7, 14, 30]       # spaced-repetition ladder (days between reviews)
MAX_HORIZON_DAYS = 180

# Knowledge graph: subject -> topic -> (intrinsic difficulty 0..1, prerequisite topics)
CURRICULUM: Dict[str, Dict[str, Tuple[float, List[str]]]] = {
    "Mathematics": {
        "Algebra & Functions": (0.4, []),
        "Limits & Continuity": (0.6, ["Algebra & Functions"]),
        "Differentiation": (0.7, ["Limits & Continuity"]),
        "Integration": (0.8, ["Differentiation"]),
    },
    "Physics": {
        "Kinematics": (0.4, []),
        "Newton's Laws": (0.6, ["Kinematics"]),
        "Work, Energy & Power": (0.6, ["Newton's Laws"]),
        "Electric Circuits": (0.7, []),
    },
    "Chemistry": {
        "Atomic Structure": (0.4, []),
        "Chemical Bonding": (0.6, ["Atomic Structure"]),
        "Stoichiometry": (0.7, ["Atomic Structure"]),
        "Acids, Bases & pH": (0.7, ["Stoichiometry"]),
    },
    "Biology": {
        "Cell Structure": (0.3, []),
        "Genetics & Inheritance": (0.6, ["Cell Structure"]),
        "Photosynthesis & Respiration": (0.7, ["Cell Structure"]),
        "Evolution & Ecology": (0.5, ["Genetics & Inheritance"]),
    },
    "Computer Science": {
        "Variables & Control Flow": (0.3, []),
        "Data Structures": (0.6, ["Variables & Control Flow"]),
        "Algorithms & Complexity": (0.8, ["Data Structures"]),
        "Databases & SQL": (0.5, ["Variables & Control Flow"]),
    },
    "History": {
        "Industrial Revolution": (0.4, []),
        "World War I": (0.5, ["Industrial Revolution"]),
        "World War II": (0.6, ["World War I"]),
        "Cold War": (0.6, ["World War II"]),
    },
}

# Offline question bank: topic -> (question, model answer, [(concept label, "kw1|kw2|...")])
QUESTION_BANK: Dict[str, Tuple[str, str, List[Tuple[str, str]]]] = {
    "Algebra & Functions": ("Explain what the domain and range of a function are, and give an example of a function with a restricted domain.",
        "Domain = all valid inputs, range = all resulting outputs. f(x)=sqrt(x) needs x >= 0; 1/x excludes x = 0.",
        [("domain is the set of valid inputs", "input|x-value|x value|allowed"), ("range is the set of outputs", "output|y-value|y value|result"), ("a restricted-domain example", "square root|sqrt|1/x|denominator|division by zero|negative")]),
    "Limits & Continuity": ("What does it mean for a function to be continuous at a point? State the conditions.",
        "f(a) is defined, the limit as x->a exists (left = right), and the limit equals f(a).",
        [("f(a) is defined", "defined|exists at"), ("the limit exists (left and right agree)", "limit exists|left and right|one-sided|limit as x"), ("the limit equals f(a)", "equal|same|= f(a)|equals f(a)")]),
    "Differentiation": ("State the chain rule and use it to differentiate sin(x^2).",
        "d/dx f(g(x)) = f'(g(x))*g'(x), so d/dx sin(x^2) = 2x*cos(x^2).",
        [("derivative of the outer times derivative of the inner", "outer|inner|multiply|times|product"), ("inner derivative 2x", "2x"), ("the cos(x^2) factor", "cos")]),
    "Integration": ("What does the Fundamental Theorem of Calculus say, and what is the integral of 2x dx?",
        "Differentiation and integration are inverse operations; a definite integral is F(b)-F(a). The integral of 2x is x^2 + C.",
        [("links derivatives and integrals as inverses", "inverse|antiderivative|opposite|connect|relationship"), ("definite integral evaluated at the bounds", "upper|lower|bounds|f(b)|difference|evaluate"), ("result x^2 + C", "x^2|x²|x squared|+ c|+c")]),
    "Kinematics": ("State the equation relating final velocity, initial velocity, acceleration and time, and say when it applies.",
        "v = u + at, valid for constant (uniform) acceleration.",
        [("v = u + at", "v = u + at|v=u+at|u + at|u+at"), ("constant acceleration only", "constant|uniform"), ("meaning of the symbols", "initial|final|velocity")]),
    "Newton's Laws": ("State Newton's second law and apply it to a 2 kg block pushed by a 10 N net force.",
        "F = ma, so a = 10/2 = 5 m/s^2 in the direction of the net force.",
        [("F = ma", "f = ma|f=ma|force equals mass|mass times acceleration|mass x acceleration"), ("uses the net force", "net force|resultant|unbalanced"), ("acceleration of 5 m/s^2", "5 m/s|5m/s|5 meters|acceleration of 5|a = 5|a=5")]),
    "Work, Energy & Power": ("Define work, kinetic energy and power, with their formulas.",
        "W = F*d*cos(theta); KE = 1/2 m v^2; P = W/t (rate of doing work, in watts).",
        [("work = force x displacement", "force times|f·d|f x d|force × distance|force x distance|w = fd|w=fd|force and displacement"), ("KE = 1/2 m v^2", "½|1/2|0.5|mv^2|mv²"), ("power = work per unit time", "per unit time|w/t|rate|watt|work divided by time")]),
    "Electric Circuits": ("State Ohm's law and explain how total resistance differs for series and parallel resistors.",
        "V = IR. Series resistances add; for parallel, 1/R = 1/R1 + 1/R2, so the total is smaller than any branch.",
        [("V = IR", "v = ir|v=ir|voltage equals|current times resistance|v = i r"), ("series resistances add", "series|add|sum"), ("parallel uses reciprocals and gives a lower total", "parallel|reciprocal|1/r|less than|decreases")]),
    "Atomic Structure": ("Describe the subatomic particles, their charges and where they are found.",
        "Protons (+) and neutrons (neutral) sit in the nucleus; electrons (-) occupy shells around it.",
        [("protons are positive", "proton|positive"), ("neutrons are neutral", "neutron|neutral|no charge"), ("electrons are negative and orbit in shells", "electron|negative|shell|orbit|cloud")]),
    "Chemical Bonding": ("Compare ionic and covalent bonding.",
        "Ionic: electrons transferred between a metal and a non-metal. Covalent: electrons shared between non-metals. Ionic compounds usually have high melting points.",
        [("ionic bonds transfer electrons", "transfer|ionic|give|donate"), ("covalent bonds share electrons", "shar|covalent"), ("a property or electronegativity contrast", "melting|conduct|electronegativity|metal|non-metal|nonmetal")]),
    "Stoichiometry": ("Explain what a mole is and how to calculate moles from mass.",
        "One mole is 6.022e23 particles; moles = mass / molar mass; balanced-equation coefficients give mole ratios.",
        [("Avogadro's number", "6.022|avogadro|10^23|10²³"), ("moles = mass / molar mass", "molar mass|mass divided|mass/|÷"), ("mole ratios from the balanced equation", "ratio|coefficient|balanced|limiting")]),
    "Acids, Bases & pH": ("Define pH and explain what pH 3 versus pH 9 tells you.",
        "pH = -log10[H+]. pH 3 is acidic (high H+), pH 9 is basic/alkaline.",
        [("pH = -log[H+]", "-log|negative log|log|h+|hydrogen ion|concentration"), ("pH 3 is acidic", "acid"), ("pH 9 is basic", "basic|alkaline|base")]),
    "Cell Structure": ("Name three organelles and state the function of each.",
        "Nucleus stores DNA; mitochondria make ATP; ribosomes build proteins.",
        [("nucleus holds DNA", "nucleus|dna|genetic"), ("mitochondria produce energy", "mitochondri|atp|energy|powerhouse"), ("a third organelle with its job", "ribosome|protein|chloroplast|membrane|golgi|endoplasmic")]),
    "Genetics & Inheritance": ("Explain dominant and recessive alleles using an Aa x Aa cross.",
        "Offspring are 1 AA : 2 Aa : 1 aa, so the dominant phenotype appears 3:1.",
        [("alleles come in pairs", "allele|gene|pair"), ("3:1 phenotype ratio", "3:1|three to one|75%|25%"), ("genotype versus phenotype", "genotype|phenotype|homozygous|heterozygous|dominant")]),
    "Photosynthesis & Respiration": ("Give the inputs and outputs of photosynthesis and of aerobic respiration.",
        "Photosynthesis: CO2 + water + light -> glucose + O2. Respiration: glucose + O2 -> CO2 + water + ATP.",
        [("photosynthesis inputs (CO2, water, light)", "carbon dioxide|co2|water|light"), ("glucose and oxygen as products or reactants", "glucose|oxygen|o2"), ("respiration releases ATP", "atp|energy|mitochondri")]),
    "Evolution & Ecology": ("Explain natural selection and give an example.",
        "Heritable variation + differential survival and reproduction under selective pressure shifts trait frequencies over generations (e.g. antibiotic resistance).",
        [("variation exists in a population", "variation|differ|diversity|mutation"), ("selective pressure favours survival", "surviv|fitness|environment|adapt|pressure"), ("traits are inherited over generations", "inherit|offspring|generation|pass|reproduc")]),
    "Variables & Control Flow": ("What is the difference between a for loop and a while loop, and when would you use each?",
        "for iterates a known sequence or count; while repeats until a condition fails, so you must update state to avoid an infinite loop.",
        [("for loops iterate a known sequence", "for|iterate|known|sequence|range"), ("while loops depend on a condition", "while|condition|until|unknown"), ("risk of infinite loops", "infinite|terminate|update|break")]),
    "Data Structures": ("Compare arrays and linked lists, and explain when a hash map is preferable.",
        "Arrays give O(1) index access; linked lists give cheap insertion via pointers; hash maps give average O(1) lookup by key.",
        [("arrays give O(1) indexed access", "index|o(1)|constant|contiguous|random access"), ("linked lists use nodes and pointers", "pointer|node|insert|link"), ("hash maps give fast key lookup", "hash|key|lookup|average o(1)")]),
    "Algorithms & Complexity": ("Explain Big-O notation and compare binary search with linear search.",
        "Big-O describes worst-case growth with input size. Binary search is O(log n) on sorted data; linear search is O(n).",
        [("Big-O describes growth with input size", "growth|upper bound|worst|scale|input size"), ("binary search is O(log n) by halving", "log n|log(n)|logarithmic|halv"), ("linear search is O(n); binary needs sorted data", "o(n)|linear|sorted")]),
    "Databases & SQL": ("Describe a SQL query that returns the three highest-paid employees, and explain what a primary key is.",
        "SELECT * FROM employees ORDER BY salary DESC LIMIT 3. A primary key uniquely identifies each row.",
        [("SELECT with ORDER BY and LIMIT", "select|order by|limit|top"), ("descending order", "desc|descending|highest"), ("primary key uniquely identifies a row", "primary key|unique|identif|row")]),
    "Industrial Revolution": ("Describe two causes and two consequences of the Industrial Revolution.",
        "Causes: coal and steam power, capital, agricultural change. Consequences: urbanisation and harsh factory conditions.",
        [("a cause such as steam, coal or capital", "steam|coal|invention|capital|agricultur|enclosure|trade"), ("urbanisation or factories", "urban|city|cities|migration|factory|factories"), ("social consequences", "working class|child labour|child labor|pollution|wages|class|conditions")]),
    "World War I": ("Explain the long-term causes of WWI and the immediate trigger.",
        "Militarism, alliances, imperialism and nationalism; the trigger was the assassination of Archduke Franz Ferdinand in Sarajevo (1914).",
        [("militarism or alliances", "militar|alliance|arms race"), ("imperialism or nationalism", "imperial|nationalis|colon"), ("the assassination of Franz Ferdinand", "assassinat|franz ferdinand|sarajevo|archduke")]),
    "World War II": ("Identify key causes of WWII and one turning point of the war.",
        "Treaty of Versailles resentment, the Depression, appeasement and Axis aggression; turning points include Stalingrad and Midway.",
        [("Versailles or the Depression", "versailles|depression|treaty|reparations|economic"), ("appeasement or aggression", "appeasement|hitler|poland|aggression|fascis|nazi"), ("a turning point", "stalingrad|midway|d-day|normandy|el alamein|turning")]),
    "Cold War": ("Define the Cold War and give two examples of proxy conflicts or crises.",
        "A US-USSR rivalry short of direct war, fought via proxy wars (Korea, Vietnam, Afghanistan) and crises (Berlin, Cuba) under nuclear deterrence.",
        [("US-USSR superpower rivalry", "soviet|ussr|united states|usa|superpower|capitalis|communis"), ("proxy wars or crises", "korea|vietnam|afghanistan|proxy|cuba"), ("nuclear deterrence or arms race", "nuclear|missile|mad|deterrence|arms race|berlin")]),
}


class Phase(str, Enum):
    ONBOARDING = "ONBOARDING"
    DIAGNOSED = "DIAGNOSED"
    PLANNED = "PLANNED"
    QUIZ_READY = "QUIZ_READY"
    EVALUATED = "EVALUATED"
    ADAPTING = "ADAPTING"


ALLOWED_TRANSITIONS: Dict[Phase, set] = {
    Phase.ONBOARDING: {Phase.DIAGNOSED},
    Phase.DIAGNOSED: {Phase.PLANNED},
    Phase.PLANNED: {Phase.ADAPTING, Phase.QUIZ_READY},
    Phase.ADAPTING: {Phase.PLANNED},
    Phase.QUIZ_READY: {Phase.EVALUATED, Phase.ADAPTING, Phase.PLANNED},
    Phase.EVALUATED: {Phase.ADAPTING, Phase.QUIZ_READY, Phase.PLANNED},
}


class SubjectInput(BaseModel):
    name: str
    confidence: int = Field(ge=1, le=5)
    quiz_score: float = Field(ge=0, le=100)
    exam_weight: int = Field(default=3, ge=1, le=5)


class StudentProfile(BaseModel):
    name: str = "Student"
    exam_date: date
    target_score: int = Field(default=85, ge=40, le=100)
    hours_per_day: float = Field(default=2.0, gt=0, le=14)
    start_hour: int = Field(default=17, ge=0, le=23)
    subjects: List[SubjectInput]


class TopicState(BaseModel):
    name: str
    subject: str
    difficulty: float
    prereqs: List[str] = Field(default_factory=list)
    weight: float = 1.0                  # exam weight / 3
    mastery: float = 0.5                 # estimated probability of answering correctly (0..1)
    sessions_needed: int = 2
    sessions_done: int = 0
    review_step: int = 0                 # index into INTERVALS
    next_review: Optional[date] = None
    last_studied: Optional[date] = None

    @property
    def learned(self) -> bool:
        return self.sessions_done >= self.sessions_needed


class StudyBlock(BaseModel):
    day_index: int
    day: date
    slot: int
    kind: str                            # learn | review | recall | mock | break | rest
    subject: str = ""
    topic: str = ""
    minutes: int = 0
    start: str = ""
    note: str = ""
    status: str = "planned"              # planned | done | missed


class StudyPlan(BaseModel):
    version: int = 1
    start: date
    exam_date: date
    blocks: List[StudyBlock] = Field(default_factory=list)
    milestones: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    changelog: List[str] = Field(default_factory=list)
    coverage: float = 0.0
    projected_mastery: float = 0.0


class PerformanceLog(BaseModel):
    ts: str
    topic: str
    subject: str
    score: float
    max_score: float = 10.0
    mastery_before: float
    mastery_after: float
    feedback: str = ""


class TraceStep(BaseModel):
    ts: str
    agent: str
    kind: str                            # THOUGHT | ACTION | OBSERVATION | DECISION | MESSAGE | STATE
    text: str


class QuizQuestion(BaseModel):
    topic: str
    subject: str
    question: str
    model_answer: str
    concepts: List[List[str]]            # [[label, "kw1|kw2"], ...]
    source: str = "offline bank"


class GradeResult(BaseModel):
    topic: str
    subject: str
    score: float
    answered: bool = True
    correct: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    misconception: str = ""
    next_step: str = ""
    model_answer: str = ""
    source: str = "rubric"


class Adjustment(BaseModel):
    topic: str
    action: str                          # reinforce | space_out | escalate
    score: float
    reason: str


class AppState(BaseModel):
    phase: Phase = Phase.ONBOARDING
    profile: Optional[StudentProfile] = None
    topics: Dict[str, TopicState] = Field(default_factory=dict)
    plan: Optional[StudyPlan] = None
    logs: List[PerformanceLog] = Field(default_factory=list)
    trace: List[TraceStep] = Field(default_factory=list)
    quiz: List[QuizQuestion] = Field(default_factory=list)
    quiz_graded: bool = False
    last_grades: List[GradeResult] = Field(default_factory=list)
    cursor: date = Field(default_factory=date.today)
    fatigue: int = 2
    calibration: Dict[str, float] = Field(default_factory=dict)
    weak: List[str] = Field(default_factory=list)
    fragile: List[str] = Field(default_factory=list)
    insight: str = ""
    baseline_mastery: float = 0.0


def add_trace(state: AppState, agent: str, kind: str, text: str) -> None:
    state.trace.append(TraceStep(ts=datetime.now().strftime("%H:%M:%S"), agent=agent, kind=kind, text=text))


def clamp(x: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, x))


def priority(t: TopicState) -> float:
    """Weighted difficulty/need score: exam weight x knowledge gap x intrinsic difficulty."""
    return t.weight * (1.0 - t.mastery) * (0.5 + t.difficulty)


def wmean(topics) -> float:
    ts = list(topics)
    tw = sum(t.weight for t in ts) or 1.0
    return sum(t.weight * t.mastery for t in ts) / tw


def apply_session(t: TopicState, kind: str, d: date) -> None:
    """Single source of truth for how a completed session changes a topic (used by simulation and real logging)."""
    t.last_studied = d
    if kind == "learn":
        t.sessions_done += 1
        t.mastery = clamp(t.mastery + 0.10, hi=0.95)
        if t.learned:
            t.review_step = 0
            t.next_review = d + timedelta(days=INTERVALS[0])
    elif kind == "review":
        t.review_step = min(t.review_step + 1, len(INTERVALS) - 1)
        t.mastery = clamp(t.mastery + 0.04, hi=0.97)
        t.next_review = d + timedelta(days=INTERVALS[t.review_step])
    elif kind == "recall":
        t.mastery = clamp(t.mastery + 0.03, hi=0.97)


class LLMClient:
    """Thin OpenAI wrapper. Every call returns None on any problem so callers fall back to rules."""
    MODEL = os.environ.get("STUDY_AGENT_MODEL", "gpt-4o-mini")

    def __init__(self) -> None:
        self._client: Any = None
        self.failures = 0
        self.last_error = ""
        self.refresh()

    def refresh(self, api_key: str = "") -> None:
        key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        self._client, self.failures = None, 0
        if key:
            os.environ["OPENAI_API_KEY"] = key
            try:
                # pyrefly: ignore [missing-import]
                from openai import OpenAI
                self._client = OpenAI(api_key=key, timeout=30)
            except Exception as exc:
                self.last_error = str(exc)[:120]

    @property
    def live(self) -> bool:
        return self._client is not None

    def json_chat(self, system: str, user: str) -> Optional[dict]:
        if not self.live:
            return None
        try:
            r = self._client.chat.completions.create(
                model=self.MODEL, temperature=0.3, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            self.failures = 0
            return json.loads(r.choices[0].message.content)
        except Exception as exc:
            self.last_error = str(exc)[:120]
            self.failures += 1
            if self.failures >= 2:      # bad key / no network: stop trying for this session
                self._client = None
            return None


LLM = LLMClient()

# %%
# Cell 4: the three agents and the orchestrator
class ReActAgent:
    """Base ReAct loop: policy() inspects working memory -> (thought, tool, args) or None to finish.
    The chosen tool runs, its observation is logged and written to memory, then the loop repeats."""
    name = "Agent"
    max_steps = 10

    def __init__(self, state: AppState, llm: LLMClient) -> None:
        self.state, self.llm = state, llm
        self.tools: Dict[str, Tuple[Callable[..., str], str]] = {}

    def log(self, kind: str, text: str) -> None:
        add_trace(self.state, self.name, kind, text)

    def policy(self, mem: Dict[str, Any]) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        raise NotImplementedError

    def loop(self, mem: Dict[str, Any]) -> Dict[str, Any]:
        self.log("STATE", "Toolbox: " + ", ".join(self.tools))
        for step in range(1, self.max_steps + 1):
            decision = self.policy(mem)
            if decision is None:
                self.log("THOUGHT", "All sub-goals satisfied, so I finish.")
                return mem
            thought, tool, args = decision
            argstr = ", ".join(f"{k}={v!r}" for k, v in args.items())
            self.log("THOUGHT", f"(step {step}) {thought}")
            self.log("ACTION", f"{tool}({argstr})")
            try:
                obs = self.tools[tool][0](mem, **args)
            except Exception as exc:
                self.log("OBSERVATION", f"Tool error {exc!r}; aborting this loop safely.")
                mem["error"] = str(exc)
                return mem
            self.log("OBSERVATION", obs)
        self.log("THOUGHT", "Step budget exhausted; stopping as a safety measure.")
        return mem


# --------------------------------------------------------------------------- Diagnostic Agent
class DiagnosticAgent(ReActAgent):
    name = "Diagnostic Agent"

    def __init__(self, state: AppState, llm: LLMClient) -> None:
        super().__init__(state, llm)
        self.tools = {
            "compute_skill_matrix": (self.t_matrix, "Blend quiz score and confidence into per-topic mastery"),
            "detect_calibration_gap": (self.t_calibration, "Compare self-confidence with measured score"),
            "apply_calibration_penalty": (self.t_penalty, "Discount mastery where the student is overconfident"),
            "propagate_prerequisites": (self.t_propagate, "Walk the knowledge graph and flag fragile foundations"),
            "rank_weak_topics": (self.t_rank, "Rank topics by weighted difficulty"),
            "generate_insight": (self.t_insight, "Write a short diagnostic summary (LLM or template)"),
        }

    def run(self) -> None:
        self.log("STATE", f"Goal: turn raw inputs into a ranked skill matrix (engine: {'LLM' if self.llm.live else 'rules'}).")
        self.loop({})

    def policy(self, mem):
        if "matrix" not in mem:
            return ("No topic-level model exists yet, so build one from subject scores and confidence.", "compute_skill_matrix", {})
        if "calibration" not in mem:
            return ("Self-assessment can be biased; compare confidence with the measured quiz score.", "detect_calibration_gap", {})
        if mem.get("overconfident") and not mem.get("penalised"):
            return (f"Overconfidence in {mem['overconfident']}: I should not trust their self-rating, so discount mastery.", "apply_calibration_penalty", {})
        if "propagated" not in mem:
            return ("A topic is only as solid as its prerequisites; propagate mastery through the graph.", "propagate_prerequisites", {})
        if "weak" not in mem:
            return ("Rank topics by weight x gap x difficulty to decide study order.", "rank_weak_topics", {})
        if "insight" not in mem:
            return ("Summarise findings for the student and the scheduler.", "generate_insight", {})
        return None

    def t_matrix(self, mem):
        st, prof = self.state, self.state.profile
        st.topics.clear()
        for s in prof.subjects:
            base = 0.6 * s.quiz_score / 100 + 0.4 * (s.confidence - 1) / 4
            for tname, (diff, pre) in CURRICULUM[s.name].items():
                jitter = (int(hashlib.md5(tname.encode()).hexdigest()[:4], 16) / 65535 - 0.5) * 0.24
                m = clamp(base + jitter - 0.15 * (diff - 0.5), 0.05, 0.95)
                need = max(1, round(1 + 3 * diff * (1 - m)))
                st.topics[tname] = TopicState(name=tname, subject=s.name, difficulty=diff, prereqs=list(pre),
                                              weight=s.exam_weight / 3, mastery=m, sessions_needed=need)
        mem["matrix"] = True
        return f"Skill matrix built: {len(st.topics)} topics, {len(prof.subjects)} subjects, mean mastery {wmean(st.topics.values()):.0%}."

    def t_calibration(self, mem):
        st = self.state
        st.calibration = {s.name: round(s.confidence / 5 - s.quiz_score / 100, 2) for s in st.profile.subjects}
        mem["calibration"] = True
        mem["overconfident"] = [k for k, v in st.calibration.items() if v > 0.25]
        under = [k for k, v in st.calibration.items() if v < -0.25]
        return f"Calibration gaps (confidence minus score): {st.calibration}. Overconfident: {mem['overconfident'] or 'none'}. Underconfident: {under or 'none'}."

    def t_penalty(self, mem):
        st, n = self.state, 0
        for t in st.topics.values():
            if t.subject in mem["overconfident"]:
                t.mastery = clamp(t.mastery - 0.06, 0.03, 0.95)
                n += 1
        mem["penalised"] = True
        return f"Lowered mastery by 6 points on {n} topics where self-rating exceeded measured skill."

    def t_propagate(self, mem):
        st, fragile = self.state, []
        for t in st.topics.values():        # CURRICULUM order guarantees prerequisites come first
            if t.prereqs:
                pm = sum(st.topics[p].mastery for p in t.prereqs) / len(t.prereqs)
                t.mastery = clamp(0.75 * t.mastery + 0.25 * pm, 0.03, 0.95)
                if pm < 0.4 and t.mastery > pm + 0.25:
                    t.mastery = clamp(t.mastery - 0.08, 0.03, 0.95)
                    fragile.append(f"{t.name} (built on weak {', '.join(t.prereqs)})")
        known = 0
        for t in st.topics.values():
            t.sessions_needed = max(1, round(1 + 3 * t.difficulty * (1 - t.mastery)))
            if t.mastery >= 0.8:           # already mastered: skip straight to spaced review
                t.sessions_done, t.review_step = t.sessions_needed, 2
                t.next_review = st.cursor + timedelta(days=INTERVALS[2])
                known += 1
        st.fragile, mem["propagated"] = fragile, True
        return f"Prerequisite propagation done. Fragile foundations: {len(fragile)}. Already-mastered topics moved to review rotation: {known}."

    def t_rank(self, mem):
        st = self.state
        ranked = sorted(st.topics.values(), key=priority, reverse=True)
        st.weak = [t.name for t in ranked[:5]]
        st.baseline_mastery = wmean(st.topics.values())
        mem["weak"] = st.weak
        return "Weakest topics: " + "; ".join(f"{t.name} ({t.mastery:.0%}, need={priority(t):.2f})" for t in ranked[:5])

    def t_insight(self, mem):
        st, prof = self.state, self.state.profile
        facts = {"weak_topics": st.weak, "fragile": st.fragile, "calibration": st.calibration,
                 "baseline_pct": round(st.baseline_mastery * 100), "target": prof.target_score,
                 "days_left": (prof.exam_date - st.cursor).days, "hours_per_day": prof.hours_per_day}
        data = self.llm.json_chat('You are a study coach. Reply ONLY with JSON {"insight": "<=70 words, concrete, encouraging"}.', json.dumps(facts))
        if data and isinstance(data.get("insight"), str):
            st.insight = data["insight"]
            src = "LLM"
        else:
            over = [k for k, v in st.calibration.items() if v > 0.25]
            gap = prof.target_score - round(st.baseline_mastery * 100)
            st.insight = (f"Start with {', '.join(st.weak[:3])}. "
                          + (f"Your confidence in {', '.join(over)} runs ahead of your quiz score, so we will test it early. " if over else "")
                          + (f"You are about {gap} points below your {prof.target_score}% target, which is closable with spaced recall." if gap > 0
                             else "You are already near your target; the plan focuses on retention."))
            src = "rule template"
        mem["insight"] = True
        return f"Insight written via {src}."


# --------------------------------------------------------------------------- Scheduler Agent
class SchedulerAgent(ReActAgent):
    name = "Scheduler Agent"
    max_steps = 16

    def __init__(self, state: AppState, llm: LLMClient) -> None:
        super().__init__(state, llm)
        self.tools = {
            "apply_forgetting_decay": (self.t_decay, "Mark missed blocks, advance the calendar, apply forgetting"),
            "adjust_for_fatigue": (self.t_fatigue, "Convert fatigue into effective daily capacity"),
            "apply_performance_adjustments": (self.t_perf, "Consume Evaluator feedback: reset or lengthen review intervals"),
            "estimate_capacity": (self.t_capacity, "Count study sessions available before the exam"),
            "spaced_repetition_allocator": (self.t_allocate, "Day-by-day allocation: due reviews, new learning, recall, mocks"),
            "validate_plan": (self.t_validate, "Check coverage and projected mastery"),
            "compress_learning": (self.t_compress, "Trade depth for coverage when the plan is infeasible"),
        }

    def run(self, trigger: str, **params: Any) -> Dict[str, Any]:
        st, prof = self.state, self.state.profile
        mem: Dict[str, Any] = {"trigger": trigger, "hours": prof.hours_per_day, "fatigue_level": st.fatigue, "notes": [], **params}
        self.log("STATE", f"Replan trigger = '{trigger}'. Cursor {st.cursor:%d %b}, exam {prof.exam_date:%d %b}, {prof.hours_per_day:g} h/day, fatigue {st.fatigue}/5.")
        self.loop(mem)
        plan = st.plan
        if plan:
            if trigger != "initial":
                plan.version += 1
            plan.changelog.append(f"v{plan.version} ({trigger}): " + "; ".join(mem["notes"][-4:]))
        return mem

    def policy(self, mem):
        if mem.get("missed_days", 0) > 0 and "decayed" not in mem:
            return (f"The student missed {mem['missed_days']} day(s). Those blocks are lost, so mark them, move the calendar and model forgetting.", "apply_forgetting_decay", {})
        if "fatigue" not in mem:
            return (f"Fatigue is {mem['fatigue_level']}/5; I must size daily workload to what the student can sustain.", "adjust_for_fatigue", {})
        if mem["trigger"] == "performance" and mem.get("adjustments") and "adjusted" not in mem:
            return (f"The Evaluator sent {len(mem['adjustments'])} adjustment(s); update spaced-repetition state before allocating.", "apply_performance_adjustments", {})
        if "capacity" not in mem:
            return ("Estimate how many sessions fit before the exam.", "estimate_capacity", {})
        if "allocated" not in mem:
            lvl = mem.get("compressed", 0)
            return ("Allocate sessions day by day: due reviews first, then new learning, interleaving subjects." + (f" Compression level {lvl} active." if lvl else ""),
                    "spaced_repetition_allocator", {})
        if "validated" not in mem:
            return ("Verify the draft plan against coverage and target mastery.", "validate_plan", {})
        if not mem["valid"] and mem.get("compressed", 0) < 2:
            return ("Plan is infeasible. Trade-off: shorten learning on strong topics rather than drop weak ones.", "compress_learning", {})
        return None

    def t_decay(self, mem):
        st, n = self.state, mem["missed_days"]
        start, end = st.cursor, st.cursor + timedelta(days=n)
        marked, topics = 0, set()
        if st.plan:
            for b in st.plan.blocks:
                if start <= b.day < end and b.status == "planned":
                    b.status = "missed"
                    if b.kind not in ("break", "rest"):
                        marked += 1
                        topics.add(b.topic)
        st.cursor = end
        for t in st.topics.values():
            if t.learned and t.last_studied:
                t.mastery = clamp(t.mastery * (1 - 0.01 * n), 0.03)
            if t.next_review and t.next_review < st.cursor:
                t.mastery = clamp(t.mastery - min(0.15, 0.02 * (st.cursor - t.next_review).days), 0.03)
        mem["decayed"] = True
        msg = f"Marked {marked} block(s) missed ({len(topics)} topics), cursor moved to {st.cursor:%a %d %b}, forgetting decay applied; overdue reviews are now due immediately."
        mem["notes"].append(f"{n} missed day(s), {marked} blocks rescheduled")
        return msg

    def t_fatigue(self, mem):
        f = int(mem["fatigue_level"])
        mult = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.75, 5: 0.6}[f]
        mem["hours_eff"] = mem["hours"] * mult
        mem["rest_days"] = {0} if f == 5 else set()
        mem["fatigue"] = True
        if mult < 1:
            mem["notes"].append(f"fatigue {f}/5 cut capacity to {mult:.0%}" + (", recovery day inserted" if f == 5 else ""))
        return f"Effective hours/day = {mem['hours']:g} x {mult:.2f} = {mem['hours_eff']:.2f}." + (" Inserting a recovery day first." if f == 5 else "")

    def t_perf(self, mem):
        st, msgs = self.state, []
        for a in mem["adjustments"]:
            t = st.topics.get(a["topic"])
            if not t:
                continue
            if a["action"] in ("reinforce", "escalate"):
                if t.learned:
                    t.review_step, t.next_review = 0, st.cursor + timedelta(days=1)
                    if a["action"] == "escalate" or a["score"] < 4:
                        t.sessions_done = max(0, t.sessions_needed - 1)
                        msgs.append(f"{t.name}: re-learn session added, review box reset")
                    else:
                        msgs.append(f"{t.name}: review pulled forward to {t.next_review:%d %b}")
                else:
                    msgs.append(f"{t.name}: not learned yet, lower mastery already raises its priority")
            elif a["action"] == "space_out" and t.learned:
                t.review_step = min(t.review_step + 1, len(INTERVALS) - 1)
                t.next_review = st.cursor + timedelta(days=INTERVALS[t.review_step])
                msgs.append(f"{t.name}: interval lengthened, next review {t.next_review:%d %b}")
        mem["adjusted"] = True
        mem["notes"].append(f"{len(msgs)} performance-driven change(s)")
        return " | ".join(msgs) or "No applicable adjustments."

    def t_capacity(self, mem):
        st, prof = self.state, self.state.profile
        days = max(0, (prof.exam_date - st.cursor).days)
        base = max(1, int(mem["hours_eff"] * 60 // (SESSION_MIN + BREAK_MIN)))
        spd = [0 if i in mem["rest_days"] else base for i in range(days)]
        if spd:
            spd[-1] = min(spd[-1], 2)       # day before the exam stays light
        mem["sessions_by_day"] = spd
        learn_demand = sum(max(0, t.sessions_needed - t.sessions_done) for t in st.topics.values() if not t.learned)
        mem["capacity"] = sum(spd)
        return f"{days} day(s) x {base} session(s) = {sum(spd)} sessions available; new-learning demand = {learn_demand} sessions (plus reviews)."

    @staticmethod
    def _pick(pool: List[TopicState], k: int, used: Dict[str, int], today: set, d: date, review: bool = False, momentum: bool = False) -> List[TopicState]:
        out: List[TopicState] = []
        for _ in range(k):
            cands = [t for t in pool if t.name not in today]
            if not cands:
                break

            def key(t: TopicState) -> float:
                bonus = 0.05 * max(0, (d - t.next_review).days) if (review and t.next_review) else 0.0
                bonus += 0.3 if (momentum and t.sessions_done > 0) else 0.0
                return priority(t) + bonus - 0.35 * used.get(t.subject, 0)     # penalty = interleave subjects
            t = max(cands, key=key)
            out.append(t)
            today.add(t.name)
            used[t.subject] = used.get(t.subject, 0) + 1
        return out

    def t_allocate(self, mem):
        st, prof = self.state, self.state.profile
        sims = {k: v.model_copy(deep=True) for k, v in st.topics.items()}
        comp = mem.get("compressed", 0)
        if comp:
            for t in sims.values():
                if not t.learned:
                    t.sessions_needed = max(t.sessions_done + 1, t.sessions_needed - comp)
        start, days, spd = st.cursor, len(mem["sessions_by_day"]), mem["sessions_by_day"]
        base_day = st.plan.start if st.plan else start
        unlearned0 = [t.name for t in sims.values() if not t.learned]
        learned_on: Dict[str, date] = {}
        blocks: List[StudyBlock] = []
        order = {"learn": 0, "review": 1, "recall": 2, "mock": 3}
        fmt = lambda m: f"{(m // 60) % 24:02d}:{m % 60:02d}"

        for di in range(days):
            d = start + timedelta(days=di)
            n = spd[di]
            day_no = (d - base_day).days + 1
            if n == 0:
                blocks.append(StudyBlock(day_index=day_no, day=d, slot=1, kind="rest", note="Recovery day: fatigue protection"))
                continue
            final_phase = (days - di) <= 2
            used: Dict[str, int] = {}
            today: set = set()
            learned_start = {t.name for t in sims.values() if t.learned}
            picks: List[Tuple[str, str, str, str]] = []     # (kind, subject, topic, note)

            if not final_phase and (di + 1) % 7 == 0 and n >= 2:
                subj_mean: Dict[str, List[float]] = {}
                for t in sims.values():
                    subj_mean.setdefault(t.subject, []).append(t.mastery)
                weakest = min(subj_mean, key=lambda s: sum(subj_mean[s]) / len(subj_mean[s]))
                picks.append(("mock", weakest, f"Mixed test: {weakest}", "Timed test. Log your result in the Quiz Hub."))

            slots = n - len(picks)
            if final_phase:
                rpool, quota = [t for t in sims.values() if t.learned], slots
            else:
                rpool = [t for t in sims.values() if t.learned and t.next_review and t.next_review <= d]
                quota = min(len(rpool), max(1, math.ceil(slots / 2))) if rpool else 0
            for t in self._pick(rpool, quota, used, today, d, review=True):
                note = ("Final revision, weakest first." if final_phase else f"Spaced-repetition box {t.review_step + 1}/{len(INTERVALS)}: retrieve from memory first, then check notes.")
                picks.append(("review", t.subject, t.name, note))
                apply_session(t, "review", d)
            slots = n - len(picks)

            learn_picks: List[Tuple[str, str, str, str, float]] = []
            while slots > 0 and not final_phase:
                unl = [t for t in sims.values() if not t.learned and t.name not in today]
                t1 = [t for t in unl if all(p in learned_start for p in t.prereqs)]
                t2 = [t for t in unl if all(p in sims and (sims[p].learned or sims[p].sessions_done > 0) for p in t.prereqs)]
                pool = t1 or t2 or unl
                if not pool:
                    break
                t = self._pick(pool, 1, used, today, d, momentum=True)[0]
                apply_session(t, "learn", d)
                note = f"Learn session {t.sessions_done}/{t.sessions_needed}" + (" (topic complete, enters spaced review)" if t.learned else "")
                if t.learned:
                    learned_on[t.name] = d
                learn_picks.append(("learn", t.subject, t.name, note, t.difficulty))
                slots -= 1
            learn_picks.sort(key=lambda p: -p[4])       # hardest topic while fresh
            picks = [p[:4] for p in learn_picks] + picks

            while slots > 0:
                rec_pool = [t for t in sims.values() if t.learned and t.name not in today]
                if not rec_pool:
                    break
                t = self._pick(rec_pool, 1, used, today, d)[0]
                picks.append(("recall", t.subject, t.name, "Blank-page recall plus 5 self-test questions."))
                apply_session(t, "recall", d)
                slots -= 1

            picks.sort(key=lambda p: order[p[0]])
            if not picks:
                blocks.append(StudyBlock(day_index=day_no, day=d, slot=1, kind="rest", note="Nothing due. Spaced repetition says you are ahead."))
                continue
            clock = prof.start_hour * 60
            for si, (kind, subj, topic, note) in enumerate(picks):
                blocks.append(StudyBlock(day_index=day_no, day=d, slot=si + 1, kind=kind, subject=subj, topic=topic,
                                         minutes=SESSION_MIN, start=fmt(clock), note=note))
                clock += SESSION_MIN
                if si < len(picks) - 1:
                    brk = LONG_BREAK_MIN if (si + 1) % 3 == 0 else BREAK_MIN
                    blocks.append(StudyBlock(day_index=day_no, day=d, slot=si + 1, kind="break", minutes=brk, start=fmt(clock),
                                             note="Walk, water, no phone"))
                    clock += brk

        # milestones
        ms: List[str] = []
        if unlearned0:
            dates = sorted(learned_on.values())
            for frac in (0.25, 0.5, 0.75, 1.0):
                k = math.ceil(len(unlearned0) * frac)
                if k <= len(dates):
                    ms.append(f"{dates[k - 1]:%a %d %b}: {k} of {len(unlearned0)} new topics learned ({int(frac * 100)}%)")
                elif frac == 1.0:
                    ms.append("Not every topic can be learned before the exam at this capacity.")
        mock_days = sorted({b.day for b in blocks if b.kind == "mock"})
        if mock_days:
            ms.append("Mock tests: " + ", ".join(f"{m:%d %b}" for m in mock_days[:6]))
        proj = wmean(sims.values())
        ms.append(f"{prof.exam_date:%a %d %b}: exam. Projected mastery {proj:.0%} (target {prof.target_score}%).")

        history = [b for b in (st.plan.blocks if st.plan else []) if b.day < st.cursor]
        if st.plan is None:
            st.plan = StudyPlan(start=start, exam_date=prof.exam_date)
        st.plan.blocks = history + blocks
        st.plan.milestones = ms
        st.plan.coverage = sum(1 for t in sims.values() if t.learned) / max(1, len(sims))
        st.plan.projected_mastery = proj
        mem.update(allocated=True, sim=sims, projected=proj)
        n_sessions = sum(1 for b in blocks if b.kind not in ("break", "rest"))
        return f"Allocated {n_sessions} sessions over {days} days. Coverage {st.plan.coverage:.0%}, projected mastery {proj:.0%}."

    def t_validate(self, mem):
        st, prof, plan = self.state, self.state.profile, self.state.plan
        unl = [t.name for t in mem["sim"].values() if not t.learned]
        proj, target = mem["projected"], prof.target_score / 100
        warns: List[str] = []
        if unl:
            warns.append(f"{len(unl)} topic(s) cannot be fully learned before the exam: {', '.join(unl[:4])}{'...' if len(unl) > 4 else ''}.")
        if proj < target:
            extra = round(max(0.25, (target - proj) * prof.hours_per_day * 1.5) * 4) / 4
            warns.append(f"Projected mastery {proj:.0%} is below your {target:.0%} target. Roughly +{extra:g} h/day would help.")
        plan.warnings = warns
        mem["valid"], mem["validated"] = not unl, True
        return ("Plan valid: every topic is learned before the exam." if not unl else f"Plan INVALID: {len(unl)} topic(s) unlearned.") + (f" Warnings: {len(warns)}." if warns else "")

    def t_compress(self, mem):
        mem["compressed"] = mem.get("compressed", 0) + 1
        mem.pop("allocated", None)
        mem.pop("validated", None)
        mem["notes"].append(f"compressed learning by {mem['compressed']} session(s) per topic")
        return f"Compression level {mem['compressed']}: each unlearned topic gets {mem['compressed']} fewer learning session(s), minimum 1. Re-allocating."


# --------------------------------------------------------------------------- Evaluator Agent
def bank_question(t: TopicState) -> QuizQuestion:
    if t.name in QUESTION_BANK:
        q, ans, concepts = QUESTION_BANK[t.name]
        return QuizQuestion(topic=t.name, subject=t.subject, question=q, model_answer=ans, concepts=[list(c) for c in concepts])
    key = t.name.lower().split()[0]
    return QuizQuestion(topic=t.name, subject=t.subject, question=f"Explain the key ideas of {t.name} and give one worked example.",
                        model_answer=f"A strong answer defines {t.name}, explains why it matters and works one example.",
                        concepts=[["a clear definition", key], ["an example", "example|for instance|e.g"]])


def heuristic_grade(q: QuizQuestion, answer: str) -> GradeResult:
    text = (answer or "").lower().strip()
    if len(text.split()) < 3:
        return GradeResult(topic=q.topic, subject=q.subject, score=0.0, answered=False, missing=[c[0] for c in q.concepts],
                           misconception="No attempt recorded.", next_step="Re-read the topic, then try to answer from memory.", model_answer=q.model_answer)
    hits = [c[0] for c in q.concepts if any(k.strip() and k.strip() in text for k in c[1].split("|"))]
    miss = [c[0] for c in q.concepts if c[0] not in hits]
    cov = len(hits) / max(1, len(q.concepts))
    depth = min(1.0, len(text.split()) / 25)
    score = round(min(10.0, 10 * cov * (0.7 + 0.3 * depth)), 1)
    if not hits:
        mis, nxt = "Foundational gap: none of the core ideas appeared.", "Re-learn the basics before more practice."
    elif miss:
        mis, nxt = f"Partial understanding: solid on {hits[0]}, but missing {miss[0]}.", f"Drill {miss[0]} with 3 flashcards, then retry."
    else:
        mis, nxt = "No misconception detected.", "Extend the interval and test with a harder variant." + (" Add detail for full marks." if depth < 1 else "")
    return GradeResult(topic=q.topic, subject=q.subject, score=score, correct=hits, missing=miss, misconception=mis, next_step=nxt, model_answer=q.model_answer)


class EvaluatorAgent(ReActAgent):
    name = "Evaluator Agent"

    def __init__(self, state: AppState, llm: LLMClient, notify: Callable[[List[Adjustment]], str]) -> None:
        super().__init__(state, llm)
        self.notify = notify
        self.tools = {
            "select_weak_topics": (self.t_select, "Choose weakest, least-recently-quizzed topics"),
            "generate_questions": (self.t_generate, "Create targeted questions (LLM or bank)"),
            "verify_questions": (self.t_verify, "Reject malformed questions and repair them"),
            "score_answers": (self.t_score, "Grade each answer against a rubric"),
            "update_mastery": (self.t_update, "Bayesian-style mastery update plus performance log"),
            "diagnose_patterns": (self.t_patterns, "Find trends and decide which adjustments to send"),
            "notify_scheduler": (self.t_notify, "Write adjustments back to the Scheduler Agent"),
        }

    def generate(self, n: int) -> List[QuizQuestion]:
        self.state.quiz, self.state.quiz_graded, self.state.last_grades = [], False, []
        self.loop({"mode": "generate", "n": n})
        return self.state.quiz

    def grade(self, answers: List[str]) -> List[GradeResult]:
        self.loop({"mode": "grade", "answers": answers})
        return self.state.last_grades

    def policy(self, mem):
        if mem["mode"] == "generate":
            if "selected" not in mem:
                return (f"Pick {mem['n']} topics where a quiz is most informative: low mastery, high weight, not just quizzed.", "select_weak_topics", {})
            if "generated" not in mem:
                return ("Write one targeted short-answer question per selected topic.", "generate_questions", {})
            if "verified" not in mem:
                return ("LLM output can be malformed; verify every question has a usable rubric.", "verify_questions", {})
            return None
        if "scored" not in mem:
            return ("Grade each submission against its concept rubric and produce step-by-step feedback.", "score_answers", {})
        if "updated" not in mem:
            return ("Convert scores into updated mastery estimates and persist them to the performance log.", "update_mastery", {})
        if "patterns" not in mem:
            return ("Look for patterns (low scores, repeated failures, strong topics) to decide whether the plan must change.", "diagnose_patterns", {})
        if mem["adjustments"] and "notified" not in mem:
            return (f"{len(mem['adjustments'])} topic(s) need schedule changes, so send an adjustment message to the Scheduler.", "notify_scheduler", {})
        if not mem["adjustments"] and "notified" not in mem:
            self.log("DECISION", "Scores are in the stable band, so no schedule change is requested.")
            mem["notified"] = True
        return None

    def t_select(self, mem):
        st = self.state
        recent = [l.topic for l in st.logs[-6:]]
        ranked = sorted(st.topics.values(), key=lambda t: priority(t) - (0.15 if t.name in recent else 0), reverse=True)
        chosen: List[TopicState] = []
        per: Dict[str, int] = {}
        for t in ranked:
            if per.get(t.subject, 0) < 2:
                chosen.append(t)
                per[t.subject] = per.get(t.subject, 0) + 1
            if len(chosen) == mem["n"]:
                break
        for t in ranked:
            if len(chosen) >= mem["n"]:
                break
            if t not in chosen:
                chosen.append(t)
        mem["topics"], mem["selected"] = [t.name for t in chosen], True
        return "Selected: " + ", ".join(f"{t.name} ({t.mastery:.0%})" for t in chosen)

    def t_generate(self, mem):
        st, made = self.state, []
        for name in mem["topics"]:
            t = st.topics[name]
            q = None
            data = self.llm.json_chat(
                'You are an exam setter. Reply ONLY with JSON {"question": str, "model_answer": str, "concepts": [{"label": str, "keywords": [str]}]} with exactly 3 concepts.',
                f"Subject: {t.subject}. Topic: {t.name}. Student mastery {t.mastery:.0%}. One short-answer question aimed at likely weaknesses.")
            if data:
                try:
                    cs = [[str(c["label"]), "|".join(str(k).lower() for k in c["keywords"])] for c in data["concepts"]]
                    q = QuizQuestion(topic=t.name, subject=t.subject, question=str(data["question"]), model_answer=str(data["model_answer"]), concepts=cs, source="LLM")
                except Exception:
                    q = None
            st.quiz.append(q or bank_question(t))
            made.append(st.quiz[-1].source)
        mem["generated"] = True
        return f"Generated {len(made)} question(s): {made.count('LLM')} by LLM, {made.count('offline bank')} from the offline bank."

    def t_verify(self, mem):
        fixed = 0
        for i, q in enumerate(self.state.quiz):
            if len(q.concepts) < 2 or len(q.question.split()) < 4:
                self.state.quiz[i] = bank_question(self.state.topics[q.topic])
                fixed += 1
        mem["verified"] = True
        return f"Verification complete, {fixed} question(s) repaired."

    def _llm_grade(self, q: QuizQuestion, a: str) -> Optional[GradeResult]:
        data = self.llm.json_chat(
            'You are a rigorous but encouraging tutor. Reply ONLY with JSON {"score": 0-10, "correct": [str], "missing": [str], "misconception": str, "next_step": str}.',
            f"Question: {q.question}\nReference answer: {q.model_answer}\nStudent answer: {a}")
        try:
            return GradeResult(topic=q.topic, subject=q.subject, score=round(max(0.0, min(10.0, float(data["score"]))), 1),
                               correct=[str(x) for x in data.get("correct", [])], missing=[str(x) for x in data.get("missing", [])],
                               misconception=str(data.get("misconception", "")), next_step=str(data.get("next_step", "")),
                               model_answer=q.model_answer, source="LLM")
        except Exception:
            return None

    def t_score(self, mem):
        st, res = self.state, []
        answers = list(mem["answers"]) + [""] * (len(st.quiz) - len(mem["answers"]))
        for q, a in zip(st.quiz, answers):
            g = self._llm_grade(q, a) if (self.llm.live and a.strip()) else None
            res.append(g or heuristic_grade(q, a))
        st.last_grades, mem["scored"] = res, True
        return "Scores: " + ", ".join(f"{g.topic} {g.score}/10" + ("" if g.answered else " (skipped)") for g in res)

    def t_update(self, mem):
        st, lines = self.state, []
        for g in st.last_grades:
            if not g.answered:
                continue
            t = st.topics[g.topic]
            before = t.mastery
            t.mastery = clamp(before + 0.4 * (g.score / 10 - before))
            st.logs.append(PerformanceLog(ts=datetime.now().strftime("%d %b %H:%M"), topic=g.topic, subject=g.subject, score=g.score,
                                          mastery_before=before, mastery_after=t.mastery, feedback=g.misconception))
            lines.append(f"{g.topic} {before:.0%} to {t.mastery:.0%}")
        mem["updated"] = True
        return "Mastery updated: " + ("; ".join(lines) or "nothing answered.")

    def t_patterns(self, mem):
        st, adjs = self.state, []
        for g in st.last_grades:
            if not g.answered:
                continue
            t = st.topics[g.topic]
            hist = [l.score for l in st.logs if l.topic == g.topic]
            if len(hist) >= 2 and hist[-1] < 6 and hist[-2] < 6:
                adjs.append(Adjustment(topic=g.topic, action="escalate", score=g.score, reason="two consecutive low scores"))
            elif g.score < 6:
                adjs.append(Adjustment(topic=g.topic, action="reinforce", score=g.score, reason=f"scored {g.score}/10"))
            elif g.score >= 8.5 and t.learned:
                adjs.append(Adjustment(topic=g.topic, action="space_out", score=g.score, reason=f"scored {g.score}/10, retention strong"))
        skipped = sum(1 for g in st.last_grades if not g.answered)
        mem["adjustments"], mem["patterns"] = [a.model_dump() for a in adjs], True
        return f"{len(adjs)} adjustment(s) proposed ({', '.join(a.action + ':' + a.topic for a in adjs) or 'none'}); skipped answers: {skipped}."

    def t_notify(self, mem):
        mem["notified"] = True
        return self.notify([Adjustment(**a) for a in mem["adjustments"]])


# --------------------------------------------------------------------------- Orchestrator (state machine + message bus)
class Orchestrator:
    def __init__(self) -> None:
        self.state = AppState()
        self.horizon = 14
        self.diag = DiagnosticAgent(self.state, LLM)
        self.scheduler = SchedulerAgent(self.state, LLM)
        self.evaluator = EvaluatorAgent(self.state, LLM, self._on_adjustments)
        self.last_feedback_note = ""

    def transition(self, new: Phase, why: str) -> bool:
        old = self.state.phase
        if new != old and new not in ALLOWED_TRANSITIONS[old]:
            add_trace(self.state, "Orchestrator", "STATE", f"Blocked illegal transition {old.value} to {new.value}.")
            return False
        if new != old:
            add_trace(self.state, "Orchestrator", "STATE", f"{old.value} to {new.value} ({why})")
        self.state.phase = new
        return True

    def onboard(self, profile: StudentProfile) -> None:
        st = self.state
        st.profile, st.cursor = profile, date.today()
        add_trace(st, "Orchestrator", "STATE", f"Student input received: {len(profile.subjects)} subjects, exam {profile.exam_date:%d %b %Y}.")
        self.diag.run()
        self.transition(Phase.DIAGNOSED, "skill matrix ready")
        add_trace(st, "Orchestrator", "MESSAGE", "Diagnostic Agent to Scheduler Agent: skill matrix, weak list, calibration gaps.")
        self.scheduler.run("initial")
        self.transition(Phase.PLANNED, "plan v1 created")

    def adapt(self, missed: int, fatigue: int, hours: float, note: str) -> str:
        st, prof = self.state, self.state.profile
        days_left = (prof.exam_date - st.cursor).days
        if days_left <= 0:
            return "The exam day has arrived. Good luck!"
        missed = int(max(0, min(missed or 0, days_left - 1)))
        fatigue = int(fatigue)
        words = (note or "").lower()
        if fatigue < 4 and any(w in words for w in ("sick", "ill", "fever", "exhaust", "burn", "tired", "stress")):
            add_trace(st, "Orchestrator", "DECISION", f"Note mentions health or stress ('{(note or '')[:40]}'), so fatigue raised from {fatigue} to 4.")
            fatigue = 4
        st.fatigue = fatigue
        if hours and hours > 0:
            prof.hours_per_day = float(hours)
        self.transition(Phase.ADAPTING, f"student reports missed={missed}, fatigue={fatigue}")
        add_trace(st, "Orchestrator", "MESSAGE", f"Student to Scheduler Agent: missed={missed}, fatigue={fatigue}, hours={prof.hours_per_day:g}, note='{note or ''}'")
        mem = self.scheduler.run("adapt", missed_days=missed)
        self.transition(Phase.PLANNED, "plan rebalanced")
        return f"**Plan v{st.plan.version} created.** " + " | ".join(mem["notes"]) + (f"\n\n{' '.join(st.plan.warnings)}" if st.plan.warnings else "")

    def complete_today(self) -> str:
        st, prof = self.state, self.state.profile
        d = st.cursor
        if d >= prof.exam_date:
            return "The exam day has arrived. Good luck!"
        n = 0
        for b in st.plan.blocks:
            if b.day == d and b.status == "planned":
                b.status = "done"
                if b.kind in ("learn", "review", "recall") and b.topic in st.topics:
                    apply_session(st.topics[b.topic], b.kind, d)
                    n += 1
        st.cursor = d + timedelta(days=1)
        add_trace(st, self.scheduler.name, "ACTION", f"log_completion({d:%d %b}): {n} session(s) applied to spaced-repetition state, cursor to {st.cursor:%d %b}.")
        return f"Logged {n} completed session(s) for {d:%a %d %b}. Mastery and review intervals updated."

    def make_quiz(self, n: int) -> List[QuizQuestion]:
        self.transition(Phase.QUIZ_READY, "quiz requested")
        return self.evaluator.generate(n)

    def submit(self, answers: List[str]) -> str:
        st = self.state
        self.transition(Phase.EVALUATED, "answers submitted")
        self.last_feedback_note = ""
        self.evaluator.grade(answers)
        st.quiz_graded = True
        return self.last_feedback_note

    def _on_adjustments(self, adjs: List[Adjustment]) -> str:
        st = self.state
        add_trace(st, "Orchestrator", "MESSAGE", "Evaluator Agent to Scheduler Agent: " + "; ".join(f"{a.action} {a.topic} ({a.reason})" for a in adjs))
        self.transition(Phase.ADAPTING, "performance feedback")
        mem = self.scheduler.run("performance", adjustments=[a.model_dump() for a in adjs])
        self.transition(Phase.PLANNED, "plan updated from performance")
        self.last_feedback_note = f"Schedule updated to **v{st.plan.version}**: " + " | ".join(mem["notes"])
        return f"Scheduler replanned (v{st.plan.version}). " + " | ".join(mem["notes"])

# %%
# Cell 5: rendering helpers (HTML and Markdown views)
SUBJECT_COLORS = {"Mathematics": "#2563eb", "Physics": "#0891b2", "Chemistry": "#16a34a", "Biology": "#65a30d",
                  "Computer Science": "#d97706", "History": "#dc2626"}
KIND_META = {"learn": ("📘", "Learn"), "review": ("🔁", "Review"), "recall": ("🧠", "Active recall"),
             "mock": ("📝", "Mock test"), "break": ("☕", "Break"), "rest": ("🌙", "Rest")}
E = html.escape


def mcolor(m: float) -> str:
    return "#dc2626" if m < 0.45 else "#d97706" if m < 0.7 else "#059669"


def cards(items: List[Tuple[str, str]]) -> str:
    return "<div class='cards'>" + "".join(f"<div class='card'><div class='v'>{v}</div><div class='k'>{E(k)}</div></div>" for k, v in items) + "</div>"


def bar(label: str, sub: str, m: float) -> str:
    return (f"<div class='barrow'><span class='bl'><i style='background:{SUBJECT_COLORS.get(sub, '#64748b')}'></i>{E(label)}</span>"
            f"<div class='bar'><div style='width:{m * 100:.0f}%;background:{mcolor(m)}'></div></div><span class='bp'>{m:.0%}</span></div>")


EMPTY = "<div class='card muted'>Nothing here yet. Build your plan in the first tab.</div>"


def render_diag(st: AppState) -> str:
    prof = st.profile
    weak_rows = "".join(f"<tr><td>{E(n)}</td><td>{E(st.topics[n].subject)}</td><td>{st.topics[n].mastery:.0%}</td><td>{priority(st.topics[n]):.2f}</td></tr>" for n in st.weak)
    cal = "".join(f"<span class='chip' style='border-color:{'#dc2626' if v > .25 else '#059669' if abs(v) <= .25 else '#d97706'}'>{E(k)}: {v:+.0%}</span>" for k, v in st.calibration.items())
    frag = "".join(f"<li>{E(f)}</li>" for f in st.fragile) or "<li>None found.</li>"
    bars = ""
    for s in prof.subjects:
        bars += f"<h4>{E(s.name)}</h4>" + "".join(bar(t.name, t.subject, t.mastery) for t in st.topics.values() if t.subject == s.name)
    return (cards([("Days to exam", str((prof.exam_date - st.cursor).days)), ("Baseline mastery", f"{st.baseline_mastery:.0%}"),
                   ("Target", f"{prof.target_score}%"), ("Hours per day", f"{prof.hours_per_day:g}")])
            + f"<div class='box'><b>Coach's read</b><p>{E(st.insight)}</p></div>"
            + f"<div class='box'><b>Priority topics</b><table><tr><th>Topic</th><th>Subject</th><th>Mastery</th><th>Need score</th></tr>{weak_rows}</table></div>"
            + f"<div class='box'><b>Confidence minus measured score</b><div>{cal}</div><b>Fragile foundations</b><ul>{frag}</ul></div>"
            + f"<div class='box'><b>Skill matrix</b>{bars}<p class='muted'>Topic-level values are inferred from your subject score and confidence, then propagated through prerequisites. Quizzes refine them.</p></div>")


def render_timeline(st: AppState) -> str:
    plan, prof = st.plan, st.profile
    by_day: Dict[date, List[StudyBlock]] = {}
    for b in plan.blocks:
        by_day.setdefault(b.day, []).append(b)
    first, cells = plan.start, []
    cells += [f"<div class='hd'>{w}</div>" for w in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")]
    cells += ["<div></div>"] * first.weekday()
    d = first
    while d <= prof.exam_date:
        cls = "day" + (" today" if d == st.cursor else "") + (" past" if d < st.cursor else "") + (" exam" if d == prof.exam_date else "")
        label = f"{d.day} {d:%b}" if (d.day == 1 or d == first) else str(d.day)
        dots = ""
        if d == prof.exam_date:
            dots = "<div class='ex'>Exam</div>"
        for b in by_day.get(d, []):
            if b.kind == "break":
                continue
            if b.kind == "rest":
                dots += "<span title='Rest day'>🌙</span>"
                continue
            c = SUBJECT_COLORS.get(b.subject, "#64748b")
            style = {"learn": f"background:{c}", "review": f"background:transparent;border:2.5px solid {c}",
                     "recall": f"background:{c}55;border:2px dotted {c}", "mock": f"background:{c};border-radius:3px;transform:rotate(45deg)"}[b.kind]
            dots += f"<span class='dot {b.status}' style='{style}' title='{E(KIND_META[b.kind][1])}: {E(b.topic)} ({b.status})'></span>"
        cells.append(f"<div class='{cls}'><div class='n'>{label}</div><div class='dots'>{dots}</div></div>")
        d += timedelta(days=1)
    legend = "".join(f"<span class='chip'><i style='background:{c}'></i>{E(s)}</span>" for s, c in SUBJECT_COLORS.items() if any(t.subject == s for t in st.topics.values()))
    key = ("<span class='chip'>● learn</span><span class='chip'>◯ spaced review</span><span class='chip'>◌ active recall</span>"
           "<span class='chip'>◆ mock test</span><span class='chip' style='border-color:#dc2626'>red outline: missed</span>")
    return f"<div class='legend'>{legend}</div><div class='legend'>{key}</div><div class='tl'>{''.join(cells)}</div>"


def render_table(st: AppState, horizon: int) -> str:
    plan, prof = st.plan, st.profile
    start = max(plan.start, st.cursor - timedelta(days=3))
    end = min(prof.exam_date, st.cursor + timedelta(days=horizon))
    by_day: Dict[date, List[StudyBlock]] = {}
    for b in plan.blocks:
        by_day.setdefault(b.day, []).append(b)
    rows = ["| Day | Date | Plan | Focus |", "|---|---|---|---|"]
    d = start
    while d <= end:
        parts, mins = [], 0
        for b in sorted(by_day.get(d, []), key=lambda x: (x.slot, x.kind == "break")):
            mark = {"done": " ✅", "missed": " ⚠️ missed", "planned": ""}[b.status]
            icon, label = KIND_META[b.kind]
            if b.kind == "break":
                parts.append(f"{icon} {b.minutes} min break")
            elif b.kind == "rest":
                parts.append(f"{icon} {b.note}")
            else:
                mins += b.minutes
                parts.append(f"`{b.start}` {icon} **{label}**: {b.subject}, {b.topic} ({b.minutes} min){mark}<br>&nbsp;&nbsp;&nbsp;<sub>{b.note}</sub>")
        if d == prof.exam_date:
            parts = ["🎯 **Exam day**"]
        day_no = (d - plan.start).days + 1
        datecell = f"**▶ {d:%a %d %b}**" if d == st.cursor else f"{d:%a %d %b}"
        rows.append(f"| {day_no} | {datecell} | {'<br>'.join(parts) or '-'} | {mins} min |")
        d += timedelta(days=1)
    return "\n".join(rows)


def render_summary(st: AppState) -> str:
    plan, prof = st.plan, st.profile
    left = sum(1 for b in plan.blocks if b.day >= st.cursor and b.kind not in ("break", "rest"))
    done = sum(1 for b in plan.blocks if b.status == "done" and b.kind not in ("break", "rest"))
    out = cards([("Days left", str(max(0, (prof.exam_date - st.cursor).days))), ("Sessions ahead", str(left)), ("Sessions done", str(done)),
                 ("Topic coverage", f"{plan.coverage:.0%}"), ("Projected mastery", f"{plan.projected_mastery:.0%}"), ("Plan version", f"v{plan.version}")])
    out += "<div class='box'><b>Milestones</b><ul>" + "".join(f"<li>{E(m)}</li>" for m in plan.milestones) + "</ul></div>"
    if plan.warnings:
        out += "<div class='box warn'><b>Heads-up</b><ul>" + "".join(f"<li>{E(w)}</li>" for w in plan.warnings) + "</ul></div>"
    out += "<div class='box'><b>Plan history</b><ul>" + "".join(f"<li>{E(c)}</li>" for c in plan.changelog[-5:]) + "</ul></div>"
    return out


def sparkline(vals: List[float]) -> str:
    if len(vals) < 2:
        return "<p class='muted'>Submit two or more quizzes to see your trend.</p>"
    w, h = 300, 70
    pts = [(5 + i * (w - 10) / (len(vals) - 1), h - 6 - (v / 10) * (h - 12)) for i, v in enumerate(vals)]
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='#0f766e'/>" for x, y in pts)
    return f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}'><path d='{path}' fill='none' stroke='#0f766e' stroke-width='2.5'/>{dots}</svg>"


def render_analytics(st: AppState) -> str:
    logs = st.logs
    avg = sum(l.score for l in logs) / len(logs) * 10 if logs else 0
    last = logs[-1].score * 10 if logs else 0
    out = cards([("Answers graded", str(len(logs))), ("Average score", f"{avg:.0f}%" if logs else "-"), ("Latest", f"{last:.0f}%" if logs else "-"),
                 ("Mastery now", f"{wmean(st.topics.values()):.0%}"), ("Since baseline", f"{(wmean(st.topics.values()) - st.baseline_mastery) * 100:+.0f} pts")])
    out += f"<div class='box'><b>Score trend (out of 10)</b>{sparkline([l.score for l in logs[-12:]])}</div>"
    subj: Dict[str, List[float]] = {}
    for t in st.topics.values():
        subj.setdefault(t.subject, []).append(t.mastery)
    out += "<div class='box'><b>Mastery by subject</b>" + "".join(bar(s, s, sum(v) / len(v)) for s, v in subj.items()) + "</div>"
    if logs:
        rows = "".join(f"<tr><td>{E(l.ts)}</td><td>{E(l.topic)}</td><td>{l.score}/10</td><td>{l.mastery_before:.0%} to {l.mastery_after:.0%}</td></tr>" for l in reversed(logs[-8:]))
        out += f"<div class='box'><b>Recent results</b><table><tr><th>When</th><th>Topic</th><th>Score</th><th>Mastery</th></tr>{rows}</table></div>"
    return out


def render_trace(o: Optional[Orchestrator]) -> str:
    mode = "OpenAI " + LLM.MODEL if LLM.live else "offline rule engine"
    if o is None or not o.state.trace:
        return f"<div class='box'>No agent activity yet. Build a plan to watch the agents think. Engine: {mode}.</div>"
    st = o.state
    rows = "".join(f"<div class='tr {s.kind}'><span class='ts'>{s.ts}</span><span class='ag'>{E(s.agent)}</span><span class='kd'>{s.kind}</span><span class='tx'>{E(s.text)}</span></div>" for s in st.trace[-400:])
    return (f"<div class='legend'><span class='chip'>State: <b>{st.phase.value}</b></span><span class='chip'>Engine: {mode}</span>"
            f"<span class='chip'>{len(st.trace)} trace steps</span></div><div class='scroll'><div>{rows}</div></div>")


def render_feedback(grades: List[GradeResult], note: str) -> str:
    out = []
    for i, g in enumerate(grades, 1):
        if not g.answered:
            out.append(f"### Q{i}: {g.topic}\nSkipped. Reference answer: {g.model_answer}\n")
            continue
        ok = "; ".join(g.correct) or "nothing yet"
        miss = "; ".join(g.missing) or "nothing"
        out.append(f"### Q{i}: {g.topic} ({g.score}/10, graded by {g.source})\n"
                   f"1. **What you got right:** {ok}\n2. **What was missing:** {miss}\n3. **Diagnosis:** {g.misconception}\n"
                   f"4. **Do next:** {g.next_step}\n\n> **Model answer:** {g.model_answer}\n")
    if note:
        out.append(f"---\n**Agent action:** {note}")
    return "\n".join(out)


def view(o: Optional[Orchestrator]) -> tuple:
    """(timeline, table, summary, analytics, trace)"""
    if o is None or o.state.plan is None:
        return (EMPTY, "_No plan yet. Start in the first tab._", EMPTY, EMPTY, render_trace(o))
    st = o.state
    return (render_timeline(st), render_table(st, o.horizon), render_summary(st), render_analytics(st), render_trace(o))

# %%
# Cell 6: Gradio UI and launch
CSS = """
.gradio-container{max-width:1180px!important;margin:auto}
.hero{background:linear-gradient(115deg,#0b3b3c,#0f766e 60%,#b45309);color:#fff;padding:22px 26px;border-radius:14px;margin-bottom:8px}
.hero h1{margin:0;font-size:1.75rem;color:#fff}.hero p{margin:6px 0 0;opacity:.92;max-width:70ch}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:10px;margin:8px 0}
.card{background:var(--background-fill-secondary,#f8fafc);border:1px solid var(--border-color-primary,#e2e8f0);border-radius:10px;padding:10px 12px}
.card .v{font-size:1.45rem;font-weight:700}.card .k{font-size:.78rem;opacity:.7}
.box{border:1px solid var(--border-color-primary,#e2e8f0);border-radius:10px;padding:10px 14px;margin:8px 0}
.box.warn{border-color:#d97706;background:rgba(217,119,6,.08)}.box table{width:100%;border-collapse:collapse}
.box td,.box th{padding:4px 6px;text-align:left;border-bottom:1px solid var(--border-color-primary,#e2e8f0);font-size:.88rem}
.muted{opacity:.65;font-size:.85rem}
.chip{display:inline-flex;align-items:center;gap:5px;border:1.5px solid var(--border-color-primary,#cbd5e1);border-radius:999px;padding:2px 10px;margin:2px 4px 2px 0;font-size:.8rem}
.chip i,.bl i{width:10px;height:10px;border-radius:50%;display:inline-block}
.legend{margin:4px 0}
.barrow{display:grid;grid-template-columns:minmax(150px,230px) 1fr 44px;gap:8px;align-items:center;margin:3px 0;font-size:.85rem}
.bl{display:flex;align-items:center;gap:6px}.bar{height:9px;border-radius:5px;background:rgba(100,116,139,.2);overflow:hidden}.bar div{height:100%}
.bp{text-align:right;font-variant-numeric:tabular-nums}
.tl{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:6px;margin-top:6px}
.tl .hd{font-size:.72rem;text-align:center;opacity:.6}
.tl .day{min-height:62px;border:1px solid var(--border-color-primary,#e2e8f0);border-radius:8px;padding:4px 6px;background:var(--background-fill-secondary,#f8fafc)}
.tl .today{outline:2.5px solid #0f766e}.tl .past{opacity:.6}.tl .exam{background:linear-gradient(135deg,#fde68a,#fdba74);color:#1f2937}
.tl .n{font-size:.72rem;opacity:.75}.dots{display:flex;flex-wrap:wrap;gap:3px;margin-top:4px}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block;box-sizing:border-box}
.dot.missed{opacity:.4;outline:1.5px solid #dc2626}.dot.done{opacity:.45}.ex{font-weight:700;font-size:.8rem}
.scroll{max-height:680px;overflow:auto;display:flex;flex-direction:column-reverse;border:1px solid var(--border-color-primary,#e2e8f0);border-radius:10px;padding:6px}
.tr{display:grid;grid-template-columns:62px 118px 100px 1fr;gap:8px;padding:5px 6px;border-bottom:1px dashed var(--border-color-primary,#e2e8f0);font-size:.82rem;line-height:1.35}
.tr .ts{opacity:.55;font-family:monospace}.tr .ag{font-weight:600}.tr .kd{font-weight:700}
.tr.THOUGHT .kd{color:#7c3aed}.tr.ACTION .kd{color:#2563eb}.tr.OBSERVATION .kd{color:#059669}
.tr.DECISION .kd{color:#d97706}.tr.MESSAGE .kd{color:#db2777}.tr.STATE .kd{color:#64748b}
@media(max-width:700px){.tr{grid-template-columns:1fr}.barrow{grid-template-columns:1fr 1fr 40px}}
"""

THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="amber", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"])

DEMO_DEFAULTS = {  # subject: (include, confidence, quiz score, exam weight)
    "Mathematics": (True, 4, 52, 5), "Physics": (True, 2, 45, 4), "Chemistry": (True, 3, 68, 3),
    "Biology": (False, 3, 60, 3), "Computer Science": (True, 4, 82, 3), "History": (False, 3, 70, 2)}


def _need_plan(o: Optional[Orchestrator]) -> bool:
    return o is None or o.state.plan is None


def build_plan(orch, api_key, name, exam_str, target, hours, start_hour, *rows):
    try:
        LLM.refresh(api_key)
        exam = date.fromisoformat(str(exam_str).strip())
        days = (exam - date.today()).days
        if days < 2 or days > MAX_HORIZON_DAYS:
            raise ValueError(f"Exam date must be between 2 and {MAX_HORIZON_DAYS} days from today (you entered {days}).")
        subs = []
        for i, sname in enumerate(CURRICULUM):
            inc, conf, score, wt = rows[4 * i: 4 * i + 4]
            if inc:
                subs.append(SubjectInput(name=sname, confidence=int(conf), quiz_score=float(score), exam_weight=int(wt)))
        if not subs:
            raise ValueError("Select at least one subject.")
        prof = StudentProfile(name=(name or "Student").strip() or "Student", exam_date=exam, target_score=int(target),
                              hours_per_day=float(hours), start_hour=int(start_hour), subjects=subs)
        o = Orchestrator()
        o.onboard(prof)
        mode = f"OpenAI {LLM.MODEL}" if LLM.live else "offline rule engine"
        msg = f"**Plan v1 is ready** for {len(o.state.topics)} topics over {days} days (engine: {mode}). Open the **Schedule** tab."
        return (o, msg, render_diag(o.state), *view(o))
    except Exception as exc:
        return (orch, f"**Could not build the plan.** {exc}", gr.update(), *[gr.update()] * 5)


def do_adapt(orch, missed, fatigue, hours, note):
    if _need_plan(orch):
        return (orch, "Build a plan first.", *view(orch))
    return (orch, orch.adapt(int(missed), int(fatigue), float(hours or 0), note), *view(orch))


def do_complete(orch):
    if _need_plan(orch):
        return (orch, "Build a plan first.", *view(orch))
    return (orch, orch.complete_today(), *view(orch))


def set_horizon(orch, horizon):
    if _need_plan(orch):
        return orch, gr.update()
    orch.horizon = int(horizon)
    return orch, render_table(orch.state, orch.horizon)


def make_quiz(orch, n):
    hidden = [gr.update(visible=False, value="")] * 6
    if _need_plan(orch):
        return (orch, "Build a plan first.", *hidden, "", *view(orch))
    n = int(n)
    qs = orch.make_quiz(n)
    md = "\n\n".join(f"**Q{i}. {q.topic}** ({q.subject}, {q.source})\n\n{q.question}" for i, q in enumerate(qs, 1))
    boxes = [gr.update(visible=i < len(qs), value="", label=f"Your answer to Q{i + 1}") for i in range(6)]
    return (orch, md, *boxes, "", *view(orch))


def submit_quiz(orch, *answers):
    if _need_plan(orch) or not orch.state.quiz:
        return (orch, "Generate a quiz first.", *view(orch))
    if orch.state.quiz_graded:
        return (orch, "This quiz is already graded. Generate a new one.", *view(orch))
    note = orch.submit(list(answers)[: len(orch.state.quiz)])
    return (orch, render_feedback(orch.state.last_grades, note), *view(orch))


def refresh_trace(orch):
    return render_trace(orch)


with gr.Blocks(theme=THEME, css=CSS, title="AI Study Planner & Performance Agent") as demo:
    orch_state = gr.State(None)
    gr.HTML("<div class='hero'><h1>AI Study Planner &amp; Performance Agent</h1>"
            "<p>Three cooperating agents diagnose your gaps, build a spaced-repetition schedule, and rewrite it whenever life or your quiz scores change.</p></div>")

    with gr.Tabs():
        with gr.Tab("Onboarding & Diagnostic"):
            with gr.Row():
                with gr.Column(scale=2):
                    api_key = gr.Textbox(label="OpenAI API key (optional)", type="password", placeholder="Leave empty to use the offline engine")
                    s_name = gr.Textbox(label="Your name", value="Alex")
                    s_exam = gr.Textbox(label="Exam date (YYYY-MM-DD)", value=(date.today() + timedelta(days=30)).isoformat())
                    s_target = gr.Slider(50, 100, value=85, step=1, label="Target score (%)")
                    s_hours = gr.Slider(0.75, 10, value=3, step=0.25, label="Study hours per day")
                    s_start = gr.Slider(5, 21, value=17, step=1, label="Daily start hour (24h clock)")
                with gr.Column(scale=3):
                    gr.Markdown("**Subjects.** Tick what you are studying, rate your confidence, and enter your latest diagnostic quiz score.")
                    subject_inputs: List[Any] = []
                    for sname, (inc, conf, score, wt) in DEMO_DEFAULTS.items():
                        with gr.Group():
                            with gr.Row():
                                c_inc = gr.Checkbox(value=inc, label=sname, scale=2)
                                c_conf = gr.Slider(1, 5, value=conf, step=1, label="Confidence (1-5)", scale=2)
                                c_score = gr.Slider(0, 100, value=score, step=1, label="Quiz score (%)", scale=2)
                                c_wt = gr.Slider(1, 5, value=wt, step=1, label="Exam weight (1-5)", scale=2)
                        subject_inputs += [c_inc, c_conf, c_score, c_wt]
            build_btn = gr.Button("Diagnose and build my plan", variant="primary", size="lg")
            build_status = gr.Markdown()
            diag_html = gr.HTML(EMPTY)

        with gr.Tab("Schedule"):
            summary_html = gr.HTML(EMPTY)
            with gr.Accordion("Life happened? Rebalance the plan", open=True):
                with gr.Row():
                    a_missed = gr.Slider(0, 14, value=0, step=1, label="Days I missed")
                    a_fatigue = gr.Slider(1, 5, value=2, step=1, label="Fatigue (1 fresh, 5 exhausted)")
                    a_hours = gr.Slider(0, 10, value=0, step=0.25, label="New hours per day (0 keeps current)")
                a_note = gr.Textbox(label="What happened? (optional)", placeholder="e.g. caught a fever and lost two days")
                with gr.Row():
                    adapt_btn = gr.Button("Adapt schedule", variant="primary")
                    done_btn = gr.Button("Complete today and advance")
                adapt_msg = gr.Markdown()
            timeline_html = gr.HTML(EMPTY)
            horizon = gr.Slider(7, 30, value=14, step=1, label="Days shown in the table")
            table_md = gr.Markdown("_No plan yet. Start in the first tab._")

        with gr.Tab("Quiz Hub"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        q_n = gr.Slider(3, 6, value=4, step=1, label="Number of questions")
                        q_btn = gr.Button("Generate quiz on my weakest topics", variant="primary")
                    quiz_md = gr.Markdown("_Generate a quiz to begin._")
                    ans_boxes = [gr.Textbox(lines=3, visible=False, label=f"Your answer to Q{i + 1}") for i in range(6)]
                    submit_btn = gr.Button("Submit answers for grading", variant="primary")
                    feedback_md = gr.Markdown()
                with gr.Column(scale=2):
                    analytics_html = gr.HTML(EMPTY)

        with gr.Tab("Agent Reasoning Trace"):
            gr.Markdown("Every thought, tool call, observation, decision and inter-agent message, in order. Newest activity is at the bottom.")
            trace_html = gr.HTML(render_trace(None))
            trace_btn = gr.Button("Refresh trace")

    VIEW = [timeline_html, table_md, summary_html, analytics_html, trace_html]
    build_btn.click(build_plan, [orch_state, api_key, s_name, s_exam, s_target, s_hours, s_start, *subject_inputs],
                    [orch_state, build_status, diag_html, *VIEW])
    adapt_btn.click(do_adapt, [orch_state, a_missed, a_fatigue, a_hours, a_note], [orch_state, adapt_msg, *VIEW])
    done_btn.click(do_complete, [orch_state], [orch_state, adapt_msg, *VIEW])
    horizon.change(set_horizon, [orch_state, horizon], [orch_state, table_md])
    q_btn.click(make_quiz, [orch_state, q_n], [orch_state, quiz_md, *ans_boxes, feedback_md, *VIEW])
    submit_btn.click(submit_quiz, [orch_state, *ans_boxes], [orch_state, feedback_md, *VIEW])
    trace_btn.click(refresh_trace, [orch_state], [trace_html])

import os
port = int(os.environ.get("PORT", 7860))
demo.queue().launch(server_name="0.0.0.0", server_port=port)