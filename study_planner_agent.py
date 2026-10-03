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

OPENAI_API_KEY = "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"   # e.g. "sk-..."  (or add a Colab Secret named OPENAI_API_KEY)

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

# B.Tech 2nd Year Term 1 Curriculum: subject -> topic -> (intrinsic difficulty 0..1, prerequisite topics)
CURRICULUM: Dict[str, Dict[str, Tuple[float, List[str]]]] = {
    "Probability and Statistics": {
        "Module I: Introduction to Statistics": (0.35, []),
        "Module II: Introduction to Probability": (0.40, ["Module I: Introduction to Statistics"]),
        "Module III: Random Variables": (0.50, ["Module II: Introduction to Probability"]),
        "Module IV: Discrete Probability Distributions": (0.55, ["Module III: Random Variables"]),
        "Module V: Continuous Probability Distributions": (0.60, ["Module III: Random Variables"]),
        "Module VI: Sampling & Estimation": (0.65, ["Module V: Continuous Probability Distributions"]),
        "Module VII: Testing of Hypothesis – I": (0.70, ["Module VI: Sampling & Estimation"]),
        "Module VIII: Testing of Hypothesis – II": (0.75, ["Module VII: Testing of Hypothesis – I"]),
        "Module IX: Correlation": (0.50, ["Module I: Introduction to Statistics"]),
        "Module X: Regression": (0.65, ["Module IX: Correlation"]),
    },
    "DSA C++": {
        "Module I: Introduction to C++ Programming": (0.35, []),
        "Module II: Control Statements": (0.40, ["Module I: Introduction to C++ Programming"]),
        "Module III: Arrays – 1D": (0.45, ["Module II: Control Statements"]),
        "Module IV: Arrays – 2D": (0.55, ["Module III: Arrays – 1D"]),
        "Module V: String Arrays": (0.50, ["Module III: Arrays – 1D"]),
        "Module VI: Structures": (0.50, ["Module III: Arrays – 1D"]),
        "Module VII: Data Structures performance Analysis": (0.65, ["Module III: Arrays – 1D"]),
        "Module VIII: Stacks": (0.60, ["Module VII: Data Structures performance Analysis"]),
        "Module IX: Queues": (0.60, ["Module VII: Data Structures performance Analysis"]),
        "Module X: STL Fundamentals & Containers": (0.70, ["Module VIII: Stacks", "Module IX: Queues"]),
    },
    "Fundamentals of Artificial Intelligence": {
        "Module - I: Introduction to Artificial Intelligence": (0.35, []),
        "Module - II: Uninformed Search Strategies": (0.50, ["Module - I: Introduction to Artificial Intelligence"]),
        "Module - III: Informed Search Strategies": (0.65, ["Module - II: Uninformed Search Strategies"]),
        "Module - IV: Optimal Decisions in Games": (0.65, ["Module - III: Informed Search Strategies"]),
        "Module - V: Inferences": (0.60, ["Module - I: Introduction to Artificial Intelligence"]),
        "Module - VI: Knowledge Representation & Reasoning": (0.60, ["Module - V: Inferences"]),
        "Module - VII: State Space Planning": (0.65, ["Module - III: Informed Search Strategies"]),
        "Module - VIII: Uncertainty in AI": (0.65, ["Module - I: Introduction to Artificial Intelligence"]),
        "Module - IX: General Model of Learning Agents": (0.55, ["Module - I: Introduction to Artificial Intelligence"]),
        "Module - X: Applications of AI": (0.40, ["Module - I: Introduction to Artificial Intelligence"]),
    },
    "Advanced DBMS": {
        "Module 1: Relational Query Languages & Extended ER Models": (0.50, []),
        "Module 2: Fundamentals of Normalization": (0.55, ["Module 1: Relational Query Languages & Extended ER Models"]),
        "Module 3: Advanced Normalization": (0.70, ["Module 2: Fundamentals of Normalization"]),
        "Module 4: Transactions in DBMS": (0.60, ["Module 1: Relational Query Languages & Extended ER Models"]),
        "Module 5: Concurrency Control": (0.70, ["Module 4: Transactions in DBMS"]),
        "Module 6: Storage and File Structure": (0.55, ["Module 1: Relational Query Languages & Extended ER Models"]),
        "Module 7: Database Recovery Techniques": (0.65, ["Module 4: Transactions in DBMS"]),
        "Module 8: Introduction to NoSQL Databases and MongoDB": (0.50, ["Module 1: Relational Query Languages & Extended ER Models"]),
        "Module 9: Performing CRUD Operations in MongoDB": (0.50, ["Module 8: Introduction to NoSQL Databases and MongoDB"]),
        "Module 10: Advanced Querying and Data Aggregation in MongoDB": (0.65, ["Module 9: Performing CRUD Operations in MongoDB"]),
    },
}

# Offline diagnostic question bank: topic -> (question, model answer, [(concept label, "kw1|kw2|...")])
QUESTION_BANK: Dict[str, Tuple[str, str, List[Tuple[str, str]]]] = {
    # --- Probability and Statistics ---
    "Module I: Introduction to Statistics": (
        "Distinguish between descriptive and inferential statistics, and define population versus sample.",
        "Descriptive statistics summarizes and describes sample data (mean, median, variance). Inferential statistics draws conclusions and predictions about an entire population based on sample data. A population is the complete group of interest, whereas a sample is a representative subset.",
        [("descriptive summarizes data", "descriptive|summarize|mean|median|variance"), ("inferential makes predictions about population", "inferential|predict|conclusion|inference|hypothesis"), ("population vs sample", "population|entire|sample|subset|representative")]
    ),
    "Module II: Introduction to Probability": (
        "State the axioms of probability and explain what makes two events mutually exclusive.",
        "Probability axioms: 1) Non-negativity P(E) >= 0. 2) Unit measure P(S) = 1 for sample space S. 3) Additivity for mutually exclusive events: P(A U B) = P(A) + P(B). Mutually exclusive events cannot occur simultaneously, meaning P(A and B) = 0.",
        [("probability between 0 and 1 / non-negative", "non-negative|>= 0|between 0 and 1|p(s)=1"), ("mutually exclusive cannot co-occur", "cannot happen together|cannot occur|disjoint|simultaneously"), ("addition rule for disjoint events", "p(a)+p(b)|p(a u b)|intersection is zero|p(a and b)=0")]
    ),
    "Module III: Random Variables": (
        "Define a random variable and contrast discrete vs continuous random variables.",
        "A random variable is a mathematical function that maps outcomes of a random phenomenon to real numbers. Discrete random variables take distinct, countable values described by a PMF. Continuous random variables take any value within a continuous interval described by a PDF.",
        [("mapping outcomes to real numbers", "function|real number|outcome|maps"), ("discrete has countable values / PMF", "discrete|countable|integers|pmf|probability mass"), ("continuous has interval values / PDF", "continuous|interval|uncountable|pdf|density")]
    ),
    "Module IV: Discrete Probability Distributions": (
        "Compare Binomial and Poisson distributions, stating parameters and the condition where Poisson approximates Binomial.",
        "Binomial models n independent Bernoulli trials with constant success probability p, parameters (n, p). Poisson models event counts over a fixed interval with parameter lambda. Poisson approximates Binomial when n is large, p is very small, and lambda = n*p is moderate.",
        [("binomial parameters n and p", "binomial|trials|success|n and p|independent"), ("poisson rate parameter lambda", "poisson|rate|interval|lambda"), ("approximation condition large n small p", "large n|small p|lambda = np|np|rare")]
    ),
    "Module V: Continuous Probability Distributions": (
        "Explain key properties of the Normal distribution and state the 68-95-99.7 empirical rule.",
        "The Normal distribution is symmetric and bell-shaped centered at mean mu with variance sigma^2. The empirical rule states that ~68% of data falls within mu +- 1*sigma, ~95% within mu +- 2*sigma, and ~99.7% within mu +- 3*sigma.",
        [("bell-shaped and symmetric about mean", "bell|symmetric|mean|gaussian|normal"), ("mean and variance parameters", "mu|sigma|standard deviation|variance"), ("68-95-99.7 empirical rule", "68|95|99.7|1 standard deviation|2 standard deviations")]
    ),
    "Module VI: Sampling & Estimation": (
        "State the Central Limit Theorem and distinguish between a point estimate and a confidence interval.",
        "Central Limit Theorem states that the distribution of sample means approaches normality as sample size n grows (n >= 30), regardless of population shape. A point estimate is a single numerical value, whereas a confidence interval provides an interval with a specified confidence level (e.g. 95%).",
        [("Central Limit Theorem normality of sample mean", "central limit|clt|sample mean|normal distribution|n >= 30"), ("point estimate is a single value", "point estimate|single value|single number|sample mean"), ("confidence interval provides a range with confidence level", "confidence interval|interval|range|margin of error|confidence level|95%")]
    ),
    "Module VII: Testing of Hypothesis – I": (
        "Define Null (H0) and Alternative (H1) hypotheses, and explain Type I vs Type II errors.",
        "Null hypothesis (H0) assumes no significant effect or status quo; Alternative hypothesis (H1) posits an effect. Type I error (alpha) occurs when a true null hypothesis is incorrectly rejected (false positive). Type II error (beta) occurs when a false null hypothesis fails to be rejected (false negative).",
        [("null vs alternative hypothesis", "null|h0|alternative|h1|no effect|difference"), ("Type I error is false positive / reject true H0", "type i|type 1|reject null when true|false positive|alpha"), ("Type II error is false negative / accept false H0", "type ii|type 2|fail to reject|false negative|beta")]
    ),
    "Module VIII: Testing of Hypothesis – II": (
        "Compare Z-test, Student's t-test, and Chi-Square test with their application criteria.",
        "Z-test applies to large samples (n >= 30) or when population variance is known. Student's t-test applies to small samples (n < 30) with unknown population variance. Chi-Square test evaluates categorical data for goodness-of-fit and test of independence.",
        [("z-test for large sample or known variance", "z-test|z test|large sample|known variance|n >= 30"), ("t-test for small sample with unknown variance", "t-test|t test|small sample|unknown variance|degrees of freedom"), ("chi-square for categorical / independence / goodness of fit", "chi-square|chi square|categorical|independence|goodness of fit")]
    ),
    "Module IX: Correlation": (
        "Define Pearson's correlation coefficient r, state its range, and explain why correlation does not imply causation.",
        "Pearson's r measures the strength and direction of linear association between two variables, ranging from -1 to +1. Correlation does not imply causation because an observed association may be driven by lurking confounding variables or reverse causality.",
        [("measures linear association", "linear|association|relationship|pearson"), ("range from -1 to +1", "-1 to 1|-1 to +1|range|negative|positive"), ("correlation vs causation", "causation|cause|confounding|lurking|does not imply")]
    ),
    "Module X: Regression": (
        "Explain Simple Linear Regression and the Ordinary Least Squares (OLS) minimization principle.",
        "Simple linear regression models y = beta0 + beta1*x + epsilon, where beta1 is the slope and beta0 is the intercept. Ordinary Least Squares (OLS) finds optimal parameters by minimizing the sum of squared vertical differences (residuals) between actual and predicted y values.",
        [("linear regression formula slope and intercept", "y =|beta0|beta1|slope|intercept|dependent|independent"), ("Ordinary Least Squares minimizes sum of squared errors", "ordinary least squares|ols|minimize|sum of squared|residuals|errors"), ("residual is observed minus predicted", "residual|predicted|observed|fitted")]
    ),

    # --- DSA C++ ---
    "Module I: Introduction to C++ Programming": (
        "Outline the four stages of C++ compilation and explain why namespaces like std are used.",
        "C++ compilation stages: Preprocessing (macro expansion, #include), Compiling (parsing to assembly), Assembling (creating object code .o/.obj), and Linking (binding libraries into an executable). Namespaces prevent naming collisions by scoping identifiers.",
        [("compilation stages preprocessing compiling linking", "preprocess|compile|assemble|link|stages"), ("role of namespace", "namespace|scope|std|collision|name conflict"), ("executable generation", "executable|object code|binary")]
    ),
    "Module II: Control Statements": (
        "Compare switch-case and if-else ladders in C++, and explain the necessity of the break statement.",
        "If-else evaluates general boolean conditions sequentially. Switch-case evaluates integral/enum values using jump tables for O(1) branch dispatch. Without break, execution falls through into subsequent case statements.",
        [("if-else evaluates boolean expressions", "if-else|if else|boolean|condition|expression"), ("switch uses integral constant / jump table", "switch|case|integral|constant|jump table|constant expression"), ("break prevents fallthrough", "break|fallthrough|terminate|exit")]
    ),
    "Module III: Arrays – 1D": (
        "How is a 1D array allocated in memory in C++, and why is pointer offset arithmetic 0-indexed?",
        "A 1D array is allocated as a contiguous block of memory. 0-indexing directly reflects pointer arithmetic: address(arr[i]) = base_address + i * sizeof(type), where index 0 has zero offset from base.",
        [("contiguous memory allocation", "contiguous|consecutive|sequential|memory"), ("0-indexed as memory offset", "offset|base address|distance|0-indexed|pointer arithmetic"), ("element address formula", "sizeof|base +|index times size")]
    ),
    "Module IV: Arrays – 2D": (
        "Explain Row-Major order storage for 2D arrays in C++ and state the element address calculation formula.",
        "C++ stores 2D arrays in Row-Major order, placing elements of row 0 consecutively, followed by row 1. Address of arr[i][j] = base + (i * total_cols + j) * sizeof(element).",
        [("C++ uses Row-Major order", "row-major|row major|c++ uses|row by row"), ("Column-Major stores column by column", "column-major|column major|column by column"), ("2D address formula", "base +|i * cols + j|row index times columns")]
    ),
    "Module V: String Arrays": (
        "Contrast C-style char arrays with C++ std::string regarding null-termination and memory safety.",
        "C-style strings are null-terminated ('\\0') char arrays with fixed buffer size, susceptible to buffer overflows. std::string is an OOP container that manages memory dynamically, tracks length explicitly, and prevents buffer overruns.",
        [("C strings are null-terminated char arrays", "null-terminated|\\0|char array|c-style"), ("std::string is dynamic and manages memory", "std::string|dynamic|size|heap|reallocate|class"), ("safety against buffer overflows", "buffer overflow|safe|bounds|length")]
    ),
    "Module VI: Structures": (
        "Explain struct in C++ and why memory structure padding / data alignment occurs.",
        "A struct groups heterogeneous data fields (public by default in C++). Compilers insert structure padding bytes between fields to align variables to natural memory word boundaries, maximizing CPU cache access speed.",
        [("user-defined type grouping heterogeneous data", "group|different types|heterogeneous|user-defined|struct"), ("public by default in C++", "public|default access|class vs struct"), ("padding and memory alignment for CPU word size", "padding|alignment|word size|offset|unused bytes|boundary")]
    ),
    "Module VII: Data Structures performance Analysis": (
        "Define Big-O, Big-Omega, and Big-Theta asymptotic notations, and compare time vs space complexity.",
        "Big-O gives an asymptotic upper bound (worst-case limit). Big-Omega gives an asymptotic lower bound (best-case). Big-Theta gives a tight bound. Time complexity measures instruction count growth with input size n; space complexity measures auxiliary memory.",
        [("Big-O is upper bound", "big-o|big o|upper bound|worst-case|asymptotic upper"), ("Big-Omega lower bound and Big-Theta tight bound", "omega|lower bound|theta|tight bound"), ("time vs space complexity", "time complexity|space complexity|input size|memory|scale")]
    ),
    "Module VIII: Stacks": (
        "Explain the LIFO principle of a Stack, its core operations with time complexity, and an application.",
        "Stack operates on Last-In First-Out (LIFO). Primary operations push(), pop(), and top() run in O(1) time. Key applications include expression parsing, parentheses matching, function call stacks, and backtracking.",
        [("LIFO principle", "lifo|last in first out|last-in"), ("push pop peek in O(1)", "push|pop|peek|top|o(1)|constant time"), ("applications like recursion or parentheses matching", "parentheses|expression|call stack|backtracking|undo")]
    ),
    "Module IX: Queues": (
        "Explain the FIFO principle of a Queue and how a Circular Queue resolves the false overflow issue.",
        "Queue operates on First-In First-Out (FIFO) with enqueue at rear and dequeue at front in O(1). In linear queues, freed front slots cannot be reused; a Circular Queue connects rear back to front using modulo arithmetic ((rear + 1) % capacity).",
        [("FIFO principle", "fifo|first in first out|first-in"), ("circular queue avoids false overflow / uses modulo", "circular queue|modulo|reusable|false overflow|wrap around|% size"), ("deque supports operations at both ends", "deque|double-ended|both ends|front and rear")]
    ),
    "Module X: STL Fundamentals & Containers": (
        "Describe the three main pillars of C++ STL and compare std::vector with std::list.",
        "STL pillars: Containers (store data), Iterators (pointer abstraction for traversal), and Algorithms (generic functions like sort, find). std::vector provides contiguous dynamic memory with O(1) random access; std::list is a doubly linked list with O(1) bidirectional node insertions.",
        [("three components containers iterators algorithms", "containers|iterators|algorithms|stl|standard template library"), ("vector dynamic array with O(1) random access", "vector|dynamic array|random access|contiguous|o(1) access"), ("list doubly linked list with O(1) insertion", "list|linked list|doubly linked|node|insertion")]
    ),

    # --- Fundamentals of Artificial Intelligence ---
    "Module - I: Introduction to Artificial Intelligence": (
        "Define an intelligent agent in AI and explain the Turing Test criterion.",
        "An intelligent agent perceives its environment via sensors and acts rationally via actuators to maximize expected performance. The Turing Test posits that a computer is intelligent if an interrogator cannot distinguish its conversational replies from a human's.",
        [("acting rationally / rational agent", "rational|agent|percept|action|goal|environment"), ("Turing test conversational indistinguishability", "turing test|imitation game|human|interrogator|indistinguishable"), ("thinking vs acting humanly vs rationally", "humanly|rationally|thinking|acting")]
    ),
    "Module - II: Uninformed Search Strategies": (
        "Compare Breadth-First Search (BFS) and Depth-First Search (DFS) on completeness, optimality, and memory.",
        "BFS expands shallowest nodes first using a FIFO queue; it is complete and optimal (for unit step costs), but requires O(b^d) exponential memory. DFS expands deepest nodes using a LIFO stack; it is not complete or optimal, but uses modest O(b*m) linear memory.",
        [("BFS complete and optimal with O(b^d) space", "bfs|breadth-first|queue|complete|optimal|exponential space|o(b^d)"), ("DFS linear space O(bm) but not optimal", "dfs|depth-first|stack|linear space|o(bm)|not optimal"), ("branching factor b and depth d", "branching factor|depth|space complexity")]
    ),
    "Module - III: Informed Search Strategies": (
        "Explain A* search evaluation function f(n) = g(n) + h(n) and the admissibility property of heuristics.",
        "A* evaluates nodes by f(n) = g(n) + h(n), where g(n) is actual cost from root to n, and h(n) is estimated heuristic cost to goal. An admissible heuristic never overestimates the actual cost (h(n) <= h*(n)), guaranteeing A* search optimality.",
        [("f(n) = g(n) + h(n) formula", "f(n)|g(n)|h(n)|actual cost|heuristic estimate"), ("admissible heuristic never overestimates", "admissible|never overestimates|underestimate|h(n) <= h*(n)|optimistic"), ("optimality of A*", "optimal|shortest path|priority queue")]
    ),
    "Module - IV: Optimal Decisions in Games": (
        "Explain the Minimax decision rule for two-player zero-sum games and how Alpha-Beta pruning improves speed.",
        "Minimax maximizes utility for MAX while anticipating MIN will choose moves minimizing MAX's payoff. Alpha-Beta pruning prunes branches that cannot influence the final decision when alpha >= beta, allowing search to twice the depth in the same time.",
        [("Minimax for zero-sum two-player games", "minimax|zero-sum|max|min|payoff|utility"), ("Alpha-Beta pruning cuts redundant branches", "alpha-beta|pruning|prune|cutoff|alpha >= beta|efficiency"), ("doubling search depth", "depth|branching factor|evaluation")]
    ),
    "Module - V: Inferences": (
        "State Modus Ponens and explain the Resolution refutation rule in propositional logic.",
        "Modus Ponens: from P and (P -> Q), infer Q. Resolution rule takes two clauses containing complementary literals (A v B) and (~A v C) and produces resolvent (B v C); proof by contradiction repeatedly applies resolution until an empty clause is derived.",
        [("Modus Ponens if P and P implies Q then Q", "modus ponens|p implies q|implication|infer q"), ("Resolution resolves complementary literals", "resolution|complementary literals|clause|resolvent|refutation"), ("proof by refutation / contradiction", "refutation|contradiction|empty clause")]
    ),
    "Module - VI: Knowledge Representation & Reasoning": (
        "Contrast Propositional Logic with First-Order Logic (FOL) regarding predicates and quantifiers.",
        "Propositional logic represents facts as atomic boolean propositions without internal structure. First-Order Logic (FOL) models the world using objects, relations (predicates), functions, and quantifiers (Universal forall, Existential exists), enabling expressiveness.",
        [("propositional logic lacks objects and relations", "propositional|facts|true or false|boolean"), ("First-Order Logic uses predicates and objects", "first-order|fol|predicates|objects|relations"), ("Universal and Existential quantifiers", "quantifiers|forall|exists|universal|existential")]
    ),
    "Module - VII: State Space Planning": (
        "Describe the STRIPS planning formalism: states, goals, preconditions, and effects.",
        "STRIPS represents states as conjunctions of function-free ground literals. Goals are desired literal states. Operators consist of Preconditions (must hold before action execution) and Effects containing Add lists (new literals) and Delete lists (retracted literals).",
        [("STRIPS components states actions goals", "strips|state|action|goal|planning"), ("Preconditions required before action", "precondition|must hold|prerequisite"), ("Add list and Delete list effects", "add list|delete list|effects|literals")]
    ),
    "Module - VIII: Uncertainty in AI": (
        "How do Bayesian Networks represent joint probability distributions compactly using DAGs?",
        "A Bayesian Network is a Directed Acyclic Graph (DAG) whose nodes represent random variables and directed arcs represent conditional dependencies. It factorizes the joint distribution as P(X1..Xn) = Product P(Xi | Parents(Xi)), exploiting conditional independence.",
        [("Directed Acyclic Graph (DAG)", "dag|directed acyclic graph|nodes|edges"), ("nodes as variables edges as dependencies", "random variables|conditional|causal|parents"), ("compact factorization product of P(Xi|Parents)", "factorization|product|parents|joint probability|compact")]
    ),
    "Module - IX: General Model of Learning Agents": (
        "Explain the roles of Critic, Learning Element, Performance Element, and Problem Generator.",
        "Performance Element executes actions based on percepts. Critic evaluates agent success against an external performance benchmark. Learning Element modifies performance components based on critic feedback. Problem Generator proposes exploratory actions.",
        [("Performance Element selects actions", "performance element|action|external|selects"), ("Critic evaluates against performance standard", "critic|evaluates|feedback|standard"), ("Learning Element improves and Problem Generator explores", "learning element|problem generator|exploration|sub-goals|improve")]
    ),
    "Module - X: Applications of AI": (
        "Discuss modern AI applications in NLP and Computer Vision, and highlight a major ethical challenge.",
        "Applications include Large Language Models and machine translation in NLP, and autonomous vehicle perception and medical radiology imaging in Computer Vision. Major ethical concerns involve algorithmic bias, privacy, hallucination risks, and accountability.",
        [("NLP application translation or LLMs", "nlp|natural language|llm|translation|chat"), ("Vision application autonomous vehicles or diagnosis", "computer vision|vision|medical|autonomous|driving|image"), ("ethical issues bias fairness safety", "bias|ethics|fairness|privacy|accountability")]
    ),

    # --- Advanced DBMS ---
    "Module 1: Relational Query Languages & Extended ER Models": (
        "Compare procedural Relational Algebra with declarative Relational Calculus, and explain EER Specialization vs Generalization.",
        "Relational Algebra specifies procedural operations (select sigma, project pi, join). Relational Calculus specifies declarative query conditions without execution sequence. In EER models, Specialization is top-down defining subclasses; Generalization is bottom-up merging entities.",
        [("Relational Algebra is procedural", "relational algebra|procedural|how|operators|sigma|project|join"), ("Relational Calculus is declarative", "relational calculus|declarative|what|tuple|domain"), ("Specialization top-down vs Generalization bottom-up", "specialization|generalization|extended er|eer|subclass|superclass|top-down|bottom-up")]
    ),
    "Module 2: Fundamentals of Normalization": (
        "Explain 1NF, 2NF, and 3NF conditions and the anomalies they eliminate.",
        "1NF mandates atomic values and eliminates repeating groups. 2NF requires 1NF and removes partial functional dependencies (all non-prime attributes fully dependent on candidate keys). 3NF requires 2NF and eliminates transitive functional dependencies.",
        [("1NF atomic values no repeating groups", "1nf|first normal form|atomic|repeating groups"), ("2NF eliminates partial dependency", "2nf|second normal form|partial dependency|full functional dependency"), ("3NF eliminates transitive dependency", "3nf|third normal form|transitive dependency|non-key")]
    ),
    "Module 3: Advanced Normalization": (
        "Define Boyce-Codd Normal Form (BCNF) and 4NF, explaining when 3NF is insufficient.",
        "BCNF requires that for every functional dependency X -> Y, X must be a superkey. 3NF is insufficient when a non-superkey determines a prime attribute. 4NF eliminates non-trivial Multi-Valued Dependencies (MVDs) where X ->-> Y exists without X being a superkey.",
        [("BCNF requires determinant X to be a superkey", "bcnf|boyce-codd|superkey|determinant|x is a superkey"), ("3NF allows prime attributes on RHS violating BCNF", "3nf allows|prime attribute|candidate key|violates bcnf"), ("4NF eliminates multivalued dependencies", "4nf|fourth normal form|multivalued|mvd|x ->-> y")]
    ),
    "Module 4: Transactions in DBMS": (
        "State and define the ACID properties of database transactions.",
        "Atomicity: all transaction steps complete or none do (all-or-nothing rollback). Consistency: preserves all database integrity constraints. Isolation: concurrent executions yield results equivalent to serial execution. Durability: committed updates persist permanently across crashes.",
        [("Atomicity all or nothing", "atomicity|all or nothing|abort|rollback"), ("Consistency preserves integrity constraints", "consistency|integrity|valid state|constraints"), ("Isolation and Durability", "isolation|concurrent|interference|durability|persist|crash|committed")]
    ),
    "Module 5: Concurrency Control": (
        "Explain Two-Phase Locking (2PL) and compare Strict 2PL vs Rigorous 2PL.",
        "2PL enforces a Growing Phase (locks acquired, none released) and Shrinking Phase (locks released, none acquired). Strict 2PL holds exclusive (X) locks until transaction completion, preventing cascading rollbacks; Rigorous 2PL holds all shared and exclusive locks until completion.",
        [("Growing phase acquires locks only", "growing phase|acquire locks|obtain locks|cannot release"), ("Shrinking phase releases locks only", "shrinking phase|release locks|cannot acquire"), ("guarantees conflict serializability", "serializability|conflict serializable|lock point")]
    ),
    "Module 6: Storage and File Structure": (
        "Compare Primary Index, Secondary Index, and Dense vs Sparse indexing in database storage.",
        "A Primary Index is ordered on the physical sorting key of the file; a Secondary Index indexes non-ordered attributes. A Dense index has an entry for every record in the data file; a Sparse index has entries only for selected blocks, reducing index size.",
        [("Primary index on ordered field", "primary index|ordered|ordering key|clustered"), ("Secondary index on non-ordered field", "secondary index|non-ordering|unclustered"), ("Dense has entry per record sparse per block", "dense|sparse|every record|every search key|block")]
    ),
    "Module 7: Database Recovery Techniques": (
        "Explain Write-Ahead Logging (WAL) and how Checkpointing accelerates system recovery.",
        "Write-Ahead Logging (WAL) requires log records describing changes to be flushed to stable storage before corresponding dirty data pages reach disk. Checkpoints periodically write all dirty buffers and a checkpoint record, bounding recovery scan time.",
        [("Write-Ahead Logging log before data write", "write-ahead|wal|log before|stable storage|redo undo"), ("Checkpoints limit recovery log scan", "checkpoint|dirty buffers|flush|recovery time|speed up"), ("Redo and Undo mechanisms", "redo|undo|committed|uncommitted")]
    ),
    "Module 8: Introduction to NoSQL Databases and MongoDB": (
        "Outline the four categories of NoSQL databases and describe MongoDB's document-oriented architecture.",
        "NoSQL categories: Document, Key-Value, Column-Family, and Graph. MongoDB stores flexible, semi-structured BSON (Binary JSON) documents inside Collections without fixed schemas, offering high horizontal scalability and dynamic indexing.",
        [("four NoSQL types document key-value column graph", "document|key-value|column|graph|nosql|mongodb|redis|cassandra"), ("MongoDB uses BSON/JSON documents in collections", "bson|json|binary json|document|collection|schema-less|flexible"), ("advantages like horizontal scaling", "scaling|sharding|horizontal|nested")]
    ),
    "Module 9: Performing CRUD Operations in MongoDB": (
        "Provide MongoDB shell commands for Create, Read, Update, and Delete operations.",
        "Create: db.users.insertOne({name: 'Alex', term: 1}) or insertMany([...]). Read: db.users.find({term: 1}). Update: db.users.updateOne({name: 'Alex'}, {$set: {score: 90}}). Delete: db.users.deleteOne({name: 'Alex'}).",
        [("insertOne and insertMany for Create", "insertone|insertmany|insert|create"), ("find and findOne for Read", "find|findone|query|read"), ("updateOne with $set and deleteOne for Update/Delete", "updateone|updatemany|$set|update|deleteone|deletemany|delete")]
    ),
    "Module 10: Advanced Querying and Data Aggregation in MongoDB": (
        "Explain MongoDB's Aggregation Pipeline and the functions of $match, $group, $project, and $sort.",
        "The Aggregation Pipeline runs documents through sequential pipeline stages. $match filters matching documents; $project includes, computes, or renames fields; $group aggregates documents using accumulator expressions ($sum, $avg); $sort reorders the resulting stream.",
        [("aggregation pipeline transforms documents through stages", "aggregation pipeline|pipeline|stages|sequential"), ("$match filters and $project selects fields", "$match|filter|$project|reshape|select"), ("$group aggregates by key with accumulators", "$group|accumulators|$sum|$avg|group by|$sort")]
    ),
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
    selected_modules: List[str] = Field(default_factory=list)


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
        key = (api_key or "").strip() or OPENAI_API_KEY.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
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
        if prof is None:
            return "No profile available."
        st.topics.clear()
        all_chosen_topics = set()
        for s in prof.subjects:
            chosen = s.selected_modules if s.selected_modules else list(CURRICULUM[s.name].keys())
            all_chosen_topics.update(chosen)

        for s in prof.subjects:
            base = 0.6 * s.quiz_score / 100 + 0.4 * (s.confidence - 1) / 4
            chosen = s.selected_modules if s.selected_modules else list(CURRICULUM[s.name].keys())
            for tname in chosen:
                if tname not in CURRICULUM[s.name]:
                    continue
                diff, pre = CURRICULUM[s.name][tname]
                jitter = (int(hashlib.md5(tname.encode()).hexdigest()[:4], 16) / 65535 - 0.5) * 0.24
                m = clamp(base + jitter - 0.15 * (diff - 0.5), 0.05, 0.95)
                need = max(1, round(1 + 3 * diff * (1 - m)))
                active_pre = [p for p in pre if p in all_chosen_topics]
                st.topics[tname] = TopicState(name=tname, subject=s.name, difficulty=diff, prereqs=active_pre,
                                              weight=s.exam_weight / 3, mastery=m, sessions_needed=need)
        mem["matrix"] = True
        return f"Skill matrix built: {len(st.topics)} topics across {len(prof.subjects)} subjects, mean mastery {wmean(st.topics.values()):.0%}."

    def t_calibration(self, mem):
        st, prof = self.state, self.state.profile
        if prof is None:
            return "No profile available."
        st.calibration = {s.name: round(s.confidence / 5 - s.quiz_score / 100, 2) for s in prof.subjects}
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
                valid_pre = [p for p in t.prereqs if p in st.topics]
                if valid_pre:
                    pm = sum(st.topics[p].mastery for p in valid_pre) / len(valid_pre)
                    t.mastery = clamp(0.75 * t.mastery + 0.25 * pm, 0.03, 0.95)
                    if pm < 0.4 and t.mastery > pm + 0.25:
                        t.mastery = clamp(t.mastery - 0.08, 0.03, 0.95)
                        fragile.append(f"{t.name} (built on weak {', '.join(valid_pre)})")
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
        if prof is None:
            return "No profile available."
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
        assert prof is not None
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
        assert prof is not None
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
        assert prof is not None
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
        prof, plan = self.state.profile, self.state.plan
        assert prof is not None and plan is not None
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
        if not data:
            return None
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
        st, prof, plan = self.state, self.state.profile, self.state.plan
        if prof is None or plan is None:
            return "Please create a study plan first."
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
        return f"**Plan v{plan.version} created.** " + " | ".join(mem["notes"]) + (f"\n\n{' '.join(plan.warnings)}" if plan.warnings else "")

    def complete_today(self) -> str:
        st, prof, plan = self.state, self.state.profile, self.state.plan
        if prof is None or plan is None:
            return "Please create a study plan first."
        d = st.cursor
        if d >= prof.exam_date:
            return "The exam day has arrived. Good luck!"
        n = 0
        for b in plan.blocks:
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
        v = st.plan.version if st.plan else 1
        self.last_feedback_note = f"Schedule updated to **v{v}**: " + " | ".join(mem["notes"])
        return f"Scheduler replanned (v{v}). " + " | ".join(mem["notes"])

# %%


# Cell 5: rendering helpers (HTML and Markdown views) - B.Tech 2nd Year Term 1
SUBJECT_COLORS = {
    "Probability and Statistics": "#38bdf8",              # Cyan / Sky
    "DSA C++": "#818cf8",                                 # Indigo / Modern Blue
    "Fundamentals of Artificial Intelligence": "#f472b6", # Rose / Pink
    "Advanced DBMS": "#fb923c",                           # Copper / Warm Amber
}

SUBJECT_ICONS = {
    "Probability and Statistics": "📊",
    "DSA C++": "💻",
    "Fundamentals of Artificial Intelligence": "🧠",
    "Advanced DBMS": "🗄️",
}

KIND_META = {
    "learn": ("📘", "Learn"),
    "review": ("🔁", "Spaced Review"),
    "recall": ("🧠", "Active Recall"),
    "mock": ("📝", "Mock Exam"),
    "break": ("☕", "Break"),
    "rest": ("🌙", "Rest Day"),
}

E = html.escape

CARD_ICONS = {
    "Days to exam": "⏳",
    "Days left": "⏳",
    "Baseline mastery": "📊",
    "Target": "🎯",
    "Hours per day": "⏱️",
    "Sessions ahead": "📅",
    "Sessions done": "✅",
    "Topic coverage": "🗺️",
    "Projected mastery": "🚀",
    "Plan version": "🏷️",
    "Answers graded": "📝",
    "Average score": "🏆",
    "Latest": "⚡",
    "Mastery now": "💡",
    "Since baseline": "📈",
}


def parse_exam_date(date_str: str) -> date:
    s = str(date_str).strip()
    try:
        return date.fromisoformat(s)
    except Exception:
        pass
    formats = [
        "%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y",
        "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d %b %Y",
        "%Y-%m-%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    raise ValueError(f"Could not parse exam date '{s}'. Please use YYYY-MM-DD (e.g. 2026-11-01) or Month DD, YYYY (e.g. Nov 01, 2026).")


def mcolor(m: float) -> str:
    if m < 0.45:
        return "linear-gradient(90deg, #ef4444, #f87171)"
    if m < 0.70:
        return "linear-gradient(90deg, #f59e0b, #fbbf24)"
    return "linear-gradient(90deg, #10b981, #38bdf8)"


def cards(items: List[Tuple[str, str]]) -> str:
    out = []
    for k, v in items:
        ico = CARD_ICONS.get(k, "📌")
        out.append(
            f"<div class='dash-card'>"
            f"  <div class='dash-card-header'><span class='dash-card-ico'>{ico}</span><span class='dash-card-k'>{E(k)}</span></div>"
            f"  <div class='dash-card-v'>{v}</div>"
            f"</div>"
        )
    return "<div class='dash-cards-grid'>" + "".join(out) + "</div>"


def bar(label: str, sub: str, m: float) -> str:
    c = SUBJECT_COLORS.get(sub, "#38bdf8")
    return (
        f"<div class='skill-bar-row'>"
        f"  <span class='skill-bar-lbl'>"
        f"    <span class='skill-dot' style='background:{c};box-shadow:0 0 8px {c}88'></span>"
        f"    <span class='skill-title'>{E(label)}</span>"
        f"  </span>"
        f"  <div class='skill-track'><div class='skill-fill' style='width:{m * 100:.0f}%;background:{mcolor(m)}'></div></div>"
        f"  <span class='skill-pct'>{m:.0%}</span>"
        f"</div>"
    )


EMPTY = (
    "<div class='dash-empty-card'>"
    "  <div class='empty-icon-wrap'>📂</div>"
    "  <div class='empty-title'>No Study Plan Synthesized Yet</div>"
    "  <div class='empty-desc'>Configure your subjects and parameters in <b>Setup &amp; Diagnostics</b>, then click <b>🚀 Diagnose &amp; Build My Plan</b>.</div>"
    "</div>"
)


def render_diag(st: AppState) -> str:
    prof = st.profile
    if prof is None:
        return EMPTY
    weak_rows = "".join(
        f"<tr>"
        f"  <td class='td-topic'><b>{E(n)}</b></td>"
        f"  <td><span class='subject-tag' style='background:{SUBJECT_COLORS.get(st.topics[n].subject, '#38bdf8')}22;color:{SUBJECT_COLORS.get(st.topics[n].subject, '#38bdf8')};border:1px solid {SUBJECT_COLORS.get(st.topics[n].subject, '#38bdf8')}44'>{E(st.topics[n].subject)}</span></td>"
        f"  <td class='td-mastery'><span class='mastery-badge'>{st.topics[n].mastery:.0%}</span></td>"
        f"  <td><span class='priority-pill'>{priority(st.topics[n]):.2f}</span></td>"
        f"</tr>"
        for n in st.weak
    )
    cal = "".join(
        f"<span class='meta-chip' style='border-color:{'#ef4444' if v > .25 else '#10b981' if abs(v) <= .25 else '#f59e0b'};background:{'rgba(239,68,68,0.12)' if v > .25 else 'rgba(16,185,129,0.12)' if abs(v) <= .25 else 'rgba(245,158,11,0.12)'};color:{'#f87171' if v > .25 else '#34d399' if abs(v) <= .25 else '#fbbf24'}'>{E(k)}: {v:+.0%}</span>"
        for k, v in st.calibration.items()
    )
    frag = "".join(f"<li class='fragile-item'>⚠️ {E(f)}</li>" for f in st.fragile) or "<li class='fragile-none'>✅ No fragile prerequisite foundations detected.</li>"
    bars = ""
    for s in prof.subjects:
        c = SUBJECT_COLORS.get(s.name, "#38bdf8")
        bars += (
            f"<div class='subject-matrix-group'>"
            f"  <div class='matrix-head'>"
            f"    <span class='matrix-bullet' style='background:{c};box-shadow:0 0 8px {c}88'></span>"
            f"    <span class='matrix-title' style='color:{c}'>{E(s.name)}</span>"
            f"    <span class='matrix-meta'>Weight: {s.exam_weight}/5 · Baseline Quiz: {s.quiz_score:.0f}%</span>"
            f"  </div>"
            + "".join(bar(t.name, t.subject, t.mastery) for t in st.topics.values() if t.subject == s.name)
            + "</div>"
        )
    return (
        cards([("Days to exam", str((prof.exam_date - st.cursor).days)), ("Baseline mastery", f"{st.baseline_mastery:.0%}"),
               ("Target", f"{prof.target_score}%"), ("Hours per day", f"{prof.hours_per_day:g} h")])
        + f"<div class='dash-panel hero-coach-panel'>"
        f"  <div class='dash-panel-title'><span class='title-ico'>💡</span><span>Cognitive Coach Assessment &amp; Strategy Brief</span></div>"
        f"  <p class='coach-body-text'>{E(st.insight)}</p>"
        f"</div>"
        + f"<div class='dash-panel'>"
        f"  <div class='dash-panel-title'><span class='title-ico'>🎯</span><span>Priority Focus Areas (Need &times; Gap &times; Exam Weight)</span></div>"
        f"  <table class='dash-table'><thead><tr><th>TOPIC</th><th>SUBJECT</th><th>CURRENT MASTERY</th><th>PRIORITY NEED</th></tr></thead><tbody>{weak_rows}</tbody></table>"
        f"</div>"
        + f"<div class='dash-panel'>"
        f"  <div class='dash-panel-title'><span class='title-ico'>⚖️</span><span>Metacognitive Calibration &amp; Foundation Risk</span></div>"
        f"  <div class='meta-subhead'>Confidence vs Measured Score Gap:</div>"
        f"  <div class='meta-chips-wrap'>{cal}</div>"
        f"  <div class='meta-subhead' style='margin-top:14px'>Prerequisite Vulnerabilities:</div>"
        f"  <ul class='dash-clean-list'>{frag}</ul>"
        f"</div>"
        + f"<div class='dash-panel'>"
        f"  <div class='dash-panel-title'><span class='title-ico'>📊</span><span>Comprehensive Competency &amp; Knowledge Matrix</span></div>"
        f"  {bars}"
        f"  <p class='dash-note'>Mastery values update dynamically via Bayesian inference after each completed quiz.</p>"
        f"</div>"
    )


def render_timeline(st: AppState) -> str:
    plan, prof = st.plan, st.profile
    if plan is None or prof is None:
        return EMPTY
    by_day: Dict[date, List[StudyBlock]] = {}
    for b in plan.blocks:
        by_day.setdefault(b.day, []).append(b)
    first, cells = plan.start, []
    cells += [f"<div class='cal-hd'>{w}</div>" for w in ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")]
    cells += ["<div class='cal-day empty-slot'></div>"] * first.weekday()
    d = first
    while d <= prof.exam_date:
        cls = "cal-day" + (" is-today" if d == st.cursor else "") + (" is-past" if d < st.cursor else "") + (" is-exam" if d == prof.exam_date else "")
        label = f"{d.day} {d:%b}" if (d.day == 1 or d == first) else str(d.day)
        dots = ""
        if d == prof.exam_date:
            dots = "<div class='exam-milestone-tag'>🎯 FINAL EXAM</div>"
        for b in by_day.get(d, []):
            if b.kind == "break":
                continue
            if b.kind == "rest":
                dots += "<span class='rest-indicator' title='Rest & Recovery'>🌙</span>"
                continue
            c = SUBJECT_COLORS.get(b.subject, "#38bdf8")
            style = {"learn": f"background:{c};box-shadow:0 0 6px {c}88",
                     "review": f"background:transparent;border:2px solid {c}",
                     "recall": f"background:{c}44;border:1.5px dashed {c}",
                     "mock": f"background:{c};border-radius:2px;transform:rotate(45deg);box-shadow:0 0 6px {c}88"}.get(b.kind, f"background:{c}")
            dots += f"<span class='session-dot {b.status}' style='{style}' title='{E(KIND_META.get(b.kind, ('', b.kind))[1])}: {E(b.topic)} ({b.status})'></span>"
        today_badge = "<span class='today-label'>TODAY</span>" if d == st.cursor else ""
        cells.append(f"<div class='{cls}'><div class='cal-day-header'><span class='day-num'>{label}</span>{today_badge}</div><div class='dots-wrap'>{dots}</div></div>")
        d += timedelta(days=1)
    legend = "".join(f"<span class='dash-chip'><i style='background:{c};box-shadow:0 0 6px {c}88'></i>{E(s)}</span>" for s, c in SUBJECT_COLORS.items() if any(t.subject == s for t in st.topics.values()))
    key = ("<span class='dash-chip'>● Learn Session</span>"
           "<span class='dash-chip'>◯ Spaced Review</span>"
           "<span class='dash-chip'>◌ Active Recall</span>"
           "<span class='dash-chip'>◆ Mock Test</span>"
           "<span class='dash-chip red-outline'>Red border: Missed</span>")
    return f"<div class='cal-legend-bar'><div class='legend-cluster'>{legend}</div><div class='legend-cluster'>{key}</div></div><div class='cal-grid'>{''.join(cells)}</div>"


def render_table(st: AppState, horizon: int) -> str:
    plan, prof = st.plan, st.profile
    if plan is None or prof is None:
        return "_No study plan generated yet. Synthesize your plan in the first tab._"
    start = max(plan.start, st.cursor - timedelta(days=3))
    end = min(prof.exam_date, st.cursor + timedelta(days=horizon))
    by_day: Dict[date, List[StudyBlock]] = {}
    for b in plan.blocks:
        by_day.setdefault(b.day, []).append(b)
    rows = ["| Day # | Calendar Date | Targeted Study Sessions | Scheduled Duration |", "|:---:|:---|:---|:---:|"]
    d = start
    while d <= end:
        parts, mins = [], 0
        for b in sorted(by_day.get(d, []), key=lambda x: (x.slot, x.kind == "break")):
            mark = {"done": " ✅ *(completed)*", "missed": " ⚠️ *(missed)*", "planned": ""}[b.status]
            icon, label = KIND_META.get(b.kind, ("📚", b.kind))
            if b.kind == "break":
                parts.append(f"{icon} {b.minutes} min break")
            elif b.kind == "rest":
                parts.append(f"{icon} **{b.note}**")
            else:
                mins += b.minutes
                parts.append(f"`{b.start}` {icon} **{label}**: {b.subject} &rarr; *{b.topic}* ({b.minutes} min){mark}<br>&nbsp;&nbsp;&nbsp;<sub>💡 {b.note}</sub>")
        if d == prof.exam_date:
            parts = ["🎯 **EXAM DAY - Final Review & Performance Readiness**"]
        day_no = (d - plan.start).days + 1
        datecell = f"**▶ {d:%a %d %b}** *(Current Day)*" if d == st.cursor else f"{d:%a %d %b}"
        rows.append(f"| Day {day_no} | {datecell} | {'<br>'.join(parts) or '-'} | **{mins} min** |")
        d += timedelta(days=1)
    return "\n".join(rows)


def render_summary(st: AppState) -> str:
    plan, prof = st.plan, st.profile
    if plan is None or prof is None:
        return EMPTY
    left = sum(1 for b in plan.blocks if b.day >= st.cursor and b.kind not in ("break", "rest"))
    done = sum(1 for b in plan.blocks if b.status == "done" and b.kind not in ("break", "rest"))
    out = cards([("Days left", str(max(0, (prof.exam_date - st.cursor).days))), ("Sessions ahead", str(left)), ("Sessions done", str(done)),
                 ("Topic coverage", f"{plan.coverage:.0%}"), ("Projected mastery", f"{plan.projected_mastery:.0%}"), ("Plan version", f"v{plan.version}")])
    ms_items = "".join(f"<li class='dash-list-item'><span class='list-bullet-ico'>🏁</span> {E(m)}</li>" for m in plan.milestones)
    out += f"<div class='dash-panel'><div class='dash-panel-title'><span class='title-ico'>🎯</span><span>Roadmap Milestones &amp; Targets</span></div><ul class='dash-clean-list'>{ms_items}</ul></div>"
    if plan.warnings:
        warn_items = "".join(f"<li class='dash-list-item warn-item'><span class='list-bullet-ico'>⚠️</span> {E(w)}</li>" for w in plan.warnings)
        out += f"<div class='dash-panel warn-panel'><div class='dash-panel-title'><span class='title-ico'>⚡</span><span>Capacity &amp; Pacing Alerts</span></div><ul class='dash-clean-list'>{warn_items}</ul></div>"
    ch_items = "".join(f"<li class='dash-list-item'><span class='log-tag'>LOG</span> {E(c)}</li>" for c in plan.changelog[-5:])
    out += f"<div class='dash-panel'><div class='dash-panel-title'><span class='title-ico'>📜</span><span>Plan Mutation History &amp; Changelog</span></div><ul class='dash-clean-list'>{ch_items}</ul></div>"
    return out


def sparkline(vals: List[float]) -> str:
    if len(vals) < 2:
        return "<p class='dash-muted' style='text-align:center;padding:16px 0'>Submit 2 or more quizzes to render your score trajectory.</p>"
    w, h = 340, 75
    pts = [(5 + i * (w - 10) / (len(vals) - 1), h - 8 - (max(0.0, min(10.0, v)) / 10) * (h - 16)) for i, v in enumerate(vals)]
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{path} L{pts[-1][0]:.1f},{h} L{pts[0][0]:.1f},{h} Z"
    dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='#38bdf8' stroke='#0d1117' stroke-width='2'/>" for x, y in pts)
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' style='overflow:visible'>"
            f"<defs><linearGradient id='sparkGrad' x1='0' y1='0' x2='0' y2='1'>"
            f"<stop offset='0%' stop-color='#38bdf8' stop-opacity='0.35'/>"
            f"<stop offset='100%' stop-color='#38bdf8' stop-opacity='0.0'/>"
            f"</linearGradient></defs>"
            f"<path d='{area}' fill='url(#sparkGrad)'/>"
            f"<path d='{path}' fill='none' stroke='#38bdf8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/>{dots}</svg>")


def render_analytics(st: AppState) -> str:
    logs = st.logs
    avg = sum(l.score for l in logs) / len(logs) * 10 if logs else 0
    last = logs[-1].score * 10 if logs else 0
    out = cards([("Answers graded", str(len(logs))), ("Average score", f"{avg:.0f}%" if logs else "-"), ("Latest", f"{last:.0f}%" if logs else "-"),
                 ("Mastery now", f"{wmean(st.topics.values()):.0%}"), ("Since baseline", f"{(wmean(st.topics.values()) - st.baseline_mastery) * 100:+.0f} pts")])
    out += f"<div class='dash-panel'><div class='dash-panel-title'><span class='title-ico'>📈</span><span>Score Trajectory (Scale 0-10)</span></div><div style='padding:6px 0'>{sparkline([l.score for l in logs[-12:]])}</div></div>"
    subj: Dict[str, List[float]] = {}
    for t in st.topics.values():
        subj.setdefault(t.subject, []).append(t.mastery)
    out += "<div class='dash-panel'><div class='dash-panel-title'><span class='title-ico'>📊</span><span>Subject Mastery Overview</span></div>" + "".join(bar(s, s, sum(v) / len(v)) for s, v in sorted(subj.items()) if v) + "</div>"
    if logs:
        rows = "".join(f"<tr><td><code>{E(l.ts)}</code></td><td style='color:#f8fafc'><b>{E(l.topic)}</b></td><td><span class='score-pill'>{l.score:.1f}/10</span></td><td><span class='mastery-shift'>{l.mastery_before:.0%} &rarr; {l.mastery_after:.0%}</span></td></tr>" for l in reversed(logs[-8:]))
        out += f"<div class='dash-panel'><div class='dash-panel-title'><span class='title-ico'>📝</span><span>Recent Evaluation Log</span></div><table class='dash-table'><thead><tr><th>Timestamp</th><th>Topic</th><th>Score</th><th>Mastery Delta</th></tr></thead><tbody>{rows}</tbody></table></div>"
    return out


def render_trace(o: Optional[Orchestrator]) -> str:
    mode = f"OpenAI {LLM.MODEL}" if LLM.live else "Offline ReAct Engine"
    if o is None or not o.state.trace:
        return (f"<div class='dash-empty-card'>"
                f"  <div class='empty-icon-wrap'>🧠</div>"
                f"  <div class='empty-title'>No Agent Activity Recorded Yet</div>"
                f"  <div class='empty-desc'>Active Engine: <code>{mode}</code>.<br>Synthesize a plan or run a quiz to observe step-by-step agentic reasoning.</div>"
                f"</div>")
    st = o.state
    rows = "".join(f"<div class='tr-row {s.kind}'><span class='tr-ts'>{s.ts}</span><span class='tr-ag'>{E(s.agent)}</span><span class='tr-kd kd-{s.kind}'>{s.kind}</span><span class='tr-tx'>{E(s.text)}</span></div>" for s in st.trace[-400:])
    return (f"<div class='terminal-wrap'>"
            f"  <div class='terminal-top-bar'>"
            f"    <div class='terminal-badges-row'>"
            f"      <span class='term-pill pulse-pill'><span class='pulse-dot-green'></span> State: <b>{st.phase.value}</b></span>"
            f"      <span class='term-pill'>Engine: <b>{mode}</b></span>"
            f"      <span class='term-pill'>Steps: <b>{len(st.trace)}</b></span>"
            f"    </div>"
            f"    <div class='terminal-sys-title'>AGENT TELEMETRY CONSOLE</div>"
            f"  </div>"
            f"  <div class='terminal-screen'><div class='terminal-scroller'>{rows}</div></div>"
            f"</div>")


def render_feedback(grades: List[GradeResult], note: str) -> str:
    out = []
    for i, g in enumerate(grades, 1):
        if not g.answered:
            out.append(f"### Q{i}: {g.topic}\n*Skipped.*\n\n> **Reference Model Answer:** {g.model_answer}\n")
            continue
        ok = "; ".join(g.correct) or "None identified"
        miss = "; ".join(g.missing) or "None"
        out.append(f"### Q{i}: {g.topic} &mdash; Score: `{g.score}/10` *(Graded via {g.source})*\n"
                   f"- **Demonstrated Concepts:** {ok}\n"
                   f"- **Missing Knowledge Gaps:** {miss}\n"
                   f"- **Diagnostic Feedback:** {g.misconception}\n"
                   f"- **Recommended Next Step:** {g.next_step}\n\n"
                   f"> **Reference Model Answer:** {g.model_answer}\n")
    if note:
        out.append(f"---\n**Orchestrator Rebalancing Note:** {note}")
    return "\n".join(out)


def view(o: Optional[Orchestrator]) -> tuple:
    """(timeline, table, summary, analytics, trace)"""
    if o is None or o.state.plan is None:
        return (EMPTY, "_No study plan generated yet. Synthesize your plan in the first tab._", EMPTY, EMPTY, render_trace(o))
    st = o.state
    return (render_timeline(st), render_table(st, o.horizon), render_summary(st), render_analytics(st), render_trace(o))


# %%
# Cell 6: B.Tech 2nd Year Term 1 Pure Dashboard Layout, Custom CSS & Gradio Blocks
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Canvas Styling */
body, .gradio-container {
  background-color: #0b0f19 !important;
  color: #f8fafc !important;
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
  max-width: 1360px !important;
  margin: auto !important;
  padding: 10px 18px !important;
}

/* Hide any footers, watermarks or branding */
footer, .gradio-footer, .built-with {
  display: none !important;
}

/* Hero Banner Card exactly matching Image */
.hero-image2 {
  background: linear-gradient(105deg, #131b2c 0%, #172338 55%, #3d2319 100%) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 18px !important;
  padding: 26px 30px !important;
  margin-bottom: 16px !important;
  box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  text-align: center !important;
  position: relative !important;
  overflow: hidden !important;
}

.hero-title-group h1 {
  font-family: 'Outfit', sans-serif !important;
  font-size: 2.25rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.01em !important;
  color: #ffffff !important;
  margin: 0 !important;
  line-height: 1.2 !important;
}

.hero-title-group p {
  color: #94a3b8 !important;
  font-size: 0.98rem !important;
  margin: 6px 0 16px !important;
  font-weight: 400 !important;
}

.hero-crests-row {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 18px !important;
  margin-bottom: 18px !important;
}

.hero-chips-row {
  display: flex !important;
  flex-wrap: wrap !important;
  justify-content: center !important;
  gap: 10px !important;
}

.hero-pill-badge {
  display: inline-flex !important;
  align-items: center !important;
  gap: 7px !important;
  background: rgba(15, 23, 42, 0.65) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  border-radius: 999px !important;
  padding: 5px 14px !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  color: #e2e8f0 !important;
}

.hero-pill-badge .dot-pink {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #f472b6;
  box-shadow: 0 0 8px #f472b6;
}

.hero-pill-badge .dot-purple {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #c084fc;
  box-shadow: 0 0 8px #c084fc;
}

.hero-pill-badge .icon-copper {
  color: #fb923c;
  font-weight: 700;
}

/* Tab Navigation matching Image */
.tabs {
  border-bottom: 1px solid #1e293b !important;
  background: transparent !important;
}

.tab-nav {
  display: flex !important;
  gap: 24px !important;
  border-bottom: 1px solid #1e293b !important;
  padding: 0 4px !important;
  margin-bottom: 18px !important;
  background: transparent !important;
}

.tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 2.5px solid transparent !important;
  color: #94a3b8 !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  padding: 10px 16px !important;
  border-radius: 0 !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

.tab-nav button:hover {
  color: #cbd5e1 !important;
}

.tab-nav button.selected {
  color: #38bdf8 !important;
  border-bottom-color: #38bdf8 !important;
  font-weight: 700 !important;
}

/* Card Header with Waves from Image */
.card-header-bar {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
  padding-bottom: 12px !important;
  margin-bottom: 16px !important;
}

.card-header-title {
  font-size: 1.15rem !important;
  font-weight: 700 !important;
  color: #f8fafc !important;
  display: flex !important;
  align-items: center !important;
  gap: 6px !important;
}

.info-icon {
  font-size: 0.88rem !important;
  color: #64748b !important;
  cursor: help !important;
}

/* Panels / Boxes in Dark Theme */
.dash-panel, .box {
  background: #121826 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 16px !important;
  padding: 18px 22px !important;
  margin: 14px 0 !important;
  color: #f8fafc !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

.dash-panel-title {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  color: #f8fafc !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
  padding-bottom: 10px !important;
  margin-bottom: 14px !important;
}

.title-ico {
  font-size: 1.25rem !important;
}

.hero-coach-panel {
  background: linear-gradient(135deg, rgba(56, 189, 248, 0.06) 0%, rgba(30, 41, 59, 0.5) 100%) !important;
  border-color: rgba(56, 189, 248, 0.25) !important;
}

.coach-body-text {
  font-size: 0.98rem !important;
  line-height: 1.65 !important;
  color: #cbd5e1 !important;
  margin: 0 !important;
}

/* Performance Snapshot Card */
.snapshot-card {
  background: #121826 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 16px !important;
  padding: 18px 20px !important;
  margin-top: 14px !important;
}

.snapshot-header {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  margin-bottom: 14px !important;
}

.snapshot-title {
  font-size: 1.08rem !important;
  font-weight: 700 !important;
  color: #f8fafc !important;
}

.snapshot-legend {
  display: flex !important;
  align-items: center !important;
  gap: 14px !important;
  font-size: 0.8rem !important;
  color: #94a3b8 !important;
}

.legend-line-blue {
  color: #38bdf8 !important;
  font-weight: 700 !important;
}

.legend-line-pink {
  color: #f472b6 !important;
  font-weight: 700 !important;
}

.snapshot-charts-row {
  display: grid !important;
  grid-template-columns: 1fr 1.6fr 1.1fr !important;
  gap: 12px !important;
  align-items: flex-end !important;
}

/* Schedule Overview Column */
.schedule-card {
  background: #121826 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 16px !important;
  padding: 18px 20px !important;
}

.schedule-header {
  font-size: 1.12rem !important;
  font-weight: 700 !important;
  color: #f8fafc !important;
  margin-bottom: 14px !important;
}

.mini-cal-weekdays {
  display: grid !important;
  grid-template-columns: repeat(7, 1fr) !important;
  text-align: center !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  color: #94a3b8 !important;
  margin-bottom: 6px !important;
}

.mini-cal-days {
  display: grid !important;
  grid-template-columns: repeat(7, 1fr) !important;
  gap: 4px !important;
  text-align: center !important;
  font-size: 0.76rem !important;
  color: #cbd5e1 !important;
  margin-bottom: 20px !important;
}

.mini-cal-days span {
  padding: 4px 0 !important;
  border-radius: 6px !important;
}

.mini-cal-days .active-blue {
  background: #0284c7 !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}

.mini-cal-days .active-cyan {
  background: #0ea5e9 !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}

.timeline-blocks-list {
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
}

.timeline-block-item {
  display: grid !important;
  grid-template-columns: 55px 1fr !important;
  gap: 10px !important;
  align-items: center !important;
}

.block-time {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.76rem !important;
  color: #94a3b8 !important;
}

.block-pill {
  padding: 10px 14px !important;
  border-radius: 10px !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 2px !important;
}

.block-blue {
  background: linear-gradient(135deg, rgba(30, 58, 138, 0.45) 0%, rgba(14, 165, 233, 0.25) 100%) !important;
  border: 1px solid rgba(56, 189, 248, 0.35) !important;
  color: #e0f2fe !important;
}

.block-copper {
  background: linear-gradient(135deg, rgba(124, 45, 18, 0.45) 0%, rgba(249, 115, 22, 0.25) 100%) !important;
  border: 1px solid rgba(251, 146, 60, 0.35) !important;
  color: #ffedd5 !important;
}

.block-sub {
  font-size: 0.72rem !important;
  opacity: 0.75 !important;
}

/* Bento Stat Cards */
.dash-cards-grid {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)) !important;
  gap: 12px !important;
  margin: 14px 0 !important;
}

.dash-card {
  background: #131b2e !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px !important;
  padding: 14px 16px !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}

.dash-card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
}

.dash-card-header {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  margin-bottom: 6px !important;
}

.dash-card-ico {
  font-size: 1.1rem !important;
}

.dash-card-k {
  font-size: 0.75rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  color: #94a3b8 !important;
}

.dash-card-v {
  font-family: 'Outfit', sans-serif !important;
  font-size: 1.7rem !important;
  font-weight: 800 !important;
  color: #f8fafc !important;
  line-height: 1.1 !important;
}

/* Dark Tables - High Contrast */
.dash-table {
  width: 100% !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  margin-top: 8px !important;
}

.dash-table th {
  background: #172238 !important;
  color: #94a3b8 !important;
  padding: 10px 14px !important;
  font-size: 0.74rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
  text-align: left !important;
}

.dash-table td {
  padding: 12px 14px !important;
  font-size: 0.9rem !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
  color: #f8fafc !important;
}

.dash-table tr:hover td {
  background: rgba(255, 255, 255, 0.02) !important;
}

.td-topic {
  color: #f8fafc !important;
  font-weight: 600 !important;
}

.subject-tag {
  display: inline-block !important;
  padding: 2px 10px !important;
  border-radius: 6px !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
}

.mastery-badge {
  color: #38bdf8 !important;
  font-weight: 700 !important;
  font-variant-numeric: tabular-nums !important;
}

.priority-pill {
  font-family: 'JetBrains Mono', monospace !important;
  background: #172238 !important;
  border: 1px solid #243048 !important;
  color: #f8fafc !important;
  border-radius: 6px !important;
  padding: 3px 8px !important;
  font-size: 0.8rem !important;
}

.score-pill {
  background: rgba(56, 189, 248, 0.15) !important;
  color: #38bdf8 !important;
  font-weight: 700 !important;
  padding: 3px 8px !important;
  border-radius: 6px !important;
}

.mastery-shift {
  color: #10b981 !important;
  font-weight: 600 !important;
}

/* Metacognitive Chips */
.meta-subhead {
  color: #94a3b8 !important;
  font-size: 0.86rem !important;
  font-weight: 600 !important;
  margin-bottom: 8px !important;
}

.meta-chips-wrap {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}

.meta-chip {
  display: inline-flex !important;
  align-items: center !important;
  border: 1.5px solid !important;
  border-radius: 999px !important;
  padding: 4px 12px !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
}

.dash-clean-list {
  list-style: none !important;
  padding: 0 !important;
  margin: 6px 0 !important;
}

.dash-clean-list li {
  padding: 6px 0 !important;
  font-size: 0.9rem !important;
  color: #e2e8f0 !important;
}

.fragile-none {
  color: #10b981 !important;
  font-weight: 600 !important;
}

.fragile-item {
  color: #fb923c !important;
}

/* Skill Bars */
.subject-matrix-group {
  margin-bottom: 16px !important;
}

.matrix-head {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  margin-bottom: 8px !important;
}

.matrix-bullet {
  width: 8px !important;
  height: 8px !important;
  border-radius: 50% !important;
}

.matrix-title {
  font-weight: 700 !important;
  font-size: 0.95rem !important;
}

.matrix-meta {
  color: #94a3b8 !important;
  font-size: 0.78rem !important;
  margin-left: 6px !important;
}

.skill-bar-row {
  display: grid !important;
  grid-template-columns: minmax(180px, 320px) 1fr 50px !important;
  gap: 12px !important;
  align-items: center !important;
  margin: 6px 0 !important;
}

.skill-bar-lbl {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
}

.skill-dot {
  width: 9px !important;
  height: 9px !important;
  border-radius: 50% !important;
}

.skill-title {
  color: #e2e8f0 !important;
  font-size: 0.85rem !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.skill-track {
  height: 9px !important;
  border-radius: 999px !important;
  background: #1c273e !important;
  overflow: hidden !important;
}

.skill-fill {
  height: 100% !important;
  border-radius: 999px !important;
}

.skill-pct {
  text-align: right !important;
  font-variant-numeric: tabular-nums !important;
  font-weight: 700 !important;
  font-size: 0.84rem !important;
  color: #94a3b8 !important;
}

.dash-note {
  color: #64748b !important;
  font-size: 0.82rem !important;
  margin-top: 10px !important;
}

/* Empty State */
.dash-empty-card {
  text-align: center !important;
  padding: 40px 20px !important;
  background: #121826 !important;
  border: 1px dashed #243048 !important;
  border-radius: 16px !important;
  margin: 16px 0 !important;
}

.empty-icon-wrap {
  font-size: 2.2rem !important;
  margin-bottom: 8px !important;
}

.empty-title {
  color: #f8fafc !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
}

.empty-desc {
  color: #94a3b8 !important;
  font-size: 0.88rem !important;
  margin-top: 4px !important;
}

/* Inputs & Form Controls */
input[type="text"], input[type="password"], textarea {
  background: #141c2e !important;
  border: 1px solid #243048 !important;
  border-radius: 8px !important;
  color: #f8fafc !important;
  font-size: 0.92rem !important;
  padding: 8px 12px !important;
}

input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
  border-color: #38bdf8 !important;
  box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
}

/* Profile and Exam Input Icons */
.profile-input-box, .exam-input-box {
  position: relative !important;
}
.profile-input-box input, .exam-input-box input {
  padding-right: 38px !important;
}
.profile-input-box::after {
  content: "👤" !important;
  position: absolute !important;
  right: 12px !important;
  bottom: 8px !important;
  font-size: 1rem !important;
  color: #64748b !important;
  pointer-events: none !important;
  z-index: 5 !important;
}
.exam-input-box::after {
  content: "📅" !important;
  position: absolute !important;
  right: 12px !important;
  bottom: 8px !important;
  font-size: 1rem !important;
  color: #64748b !important;
  pointer-events: none !important;
  z-index: 5 !important;
}

/* Custom Checkbox as Cyan Pill Toggle Switch from Image */
input[type="checkbox"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 44px !important;
  height: 24px !important;
  border-radius: 999px !important;
  background: #1e293b !important;
  position: relative !important;
  cursor: pointer !important;
  outline: none !important;
  border: 1px solid #334155 !important;
  transition: all 0.25s ease !important;
  flex-shrink: 0 !important;
  margin-right: 8px !important;
}

input[type="checkbox"]:checked {
  background: #38bdf8 !important;
  border-color: #38bdf8 !important;
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.45) !important;
}

input[type="checkbox"]::after {
  content: "" !important;
  position: absolute !important;
  top: 3px !important;
  left: 3px !important;
  width: 16px !important;
  height: 16px !important;
  border-radius: 50% !important;
  background: #ffffff !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

input[type="checkbox"]:checked::after {
  transform: translateX(20px) !important;
}

/* Sliders */
input[type="range"] {
  accent-color: #38bdf8 !important;
}

/* Buttons */
button.primary, button[variant="primary"] {
  background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 10px 18px !important;
  box-shadow: 0 4px 14px rgba(56, 189, 248, 0.35) !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

button.primary:hover, button[variant="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
}

button.secondary, button[variant="secondary"] {
  background: #172238 !important;
  border: 1px solid #2a3754 !important;
  color: #f8fafc !important;
  border-radius: 10px !important;
  padding: 10px 18px !important;
}

/* Calendar Timeline */
.cal-grid {
  display: grid !important;
  grid-template-columns: repeat(7, minmax(0, 1fr)) !important;
  gap: 8px !important;
  margin-top: 10px !important;
}

.cal-hd {
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  text-align: center !important;
  color: #94a3b8 !important;
  letter-spacing: 0.06em !important;
  padding: 4px 0 !important;
}

.cal-day {
  min-height: 72px !important;
  border: 1px solid #1c273e !important;
  border-radius: 10px !important;
  padding: 6px 8px !important;
  background: #131b2e !important;
  display: flex !important;
  flex-direction: column !important;
  transition: border-color 0.2s ease !important;
}

.cal-day:hover {
  border-color: #38bdf8 !important;
}

.cal-day.empty-slot {
  background: transparent !important;
  border-color: transparent !important;
}

.cal-day.is-today {
  border: 2px solid #38bdf8 !important;
  background: rgba(56, 189, 248, 0.05) !important;
  box-shadow: 0 0 16px rgba(56, 189, 248, 0.3) !important;
}

.cal-day.is-past {
  opacity: 0.5 !important;
}

.cal-day.is-exam {
  background: linear-gradient(135deg, #f59e0b, #ec4899) !important;
  color: #ffffff !important;
  border-color: transparent !important;
  box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4) !important;
}

.cal-day-header {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  margin-bottom: 4px !important;
}

.day-num {
  font-size: 0.76rem !important;
  font-weight: 700 !important;
  color: #94a3b8 !important;
}

.cal-day.is-exam .day-num {
  color: #ffffff !important;
}

.today-label {
  font-size: 0.6rem !important;
  font-weight: 800 !important;
  background: #38bdf8 !important;
  color: #0b0f19 !important;
  padding: 1px 4px !important;
  border-radius: 3px !important;
}

.exam-milestone-tag {
  font-size: 0.7rem !important;
  font-weight: 800 !important;
  background: rgba(0, 0, 0, 0.3) !important;
  padding: 2px 4px !important;
  border-radius: 4px !important;
  text-align: center !important;
  margin-top: 4px !important;
}

.dots-wrap {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 3px !important;
  margin-top: auto !important;
}

.session-dot {
  width: 10px !important;
  height: 10px !important;
  border-radius: 50% !important;
  display: inline-block !important;
}

.session-dot.missed {
  opacity: 0.45 !important;
  outline: 2px solid #ef4444 !important;
}

.session-dot.done {
  opacity: 0.4 !important;
}

.rest-indicator {
  font-size: 0.8rem !important;
}

.cal-legend-bar {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 10px !important;
  align-items: center !important;
  justify-content: space-between !important;
  background: #131b2e !important;
  padding: 10px 14px !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  margin-bottom: 12px !important;
}

.legend-cluster {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 6px !important;
}

.dash-chip {
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
  background: #172238 !important;
  border: 1px solid #243048 !important;
  color: #cbd5e1 !important;
  border-radius: 999px !important;
  padding: 3px 10px !important;
  font-size: 0.76rem !important;
  font-weight: 500 !important;
}

.dash-chip i {
  width: 8px !important;
  height: 8px !important;
  border-radius: 50% !important;
  display: inline-block !important;
}

.dash-chip.red-outline {
  border-color: #ef4444 !important;
  color: #f87171 !important;
}

/* Agent Telemetry Terminal */
.terminal-wrap {
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  background: #080c14 !important;
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.45) !important;
}

.terminal-top-bar {
  background: #0d131f !important;
  padding: 12px 18px !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
}

.terminal-sys-title {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  color: #94a3b8 !important;
}

.terminal-badges-row {
  display: flex !important;
  gap: 8px !important;
}

.term-pill {
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #f8fafc !important;
  font-size: 0.74rem !important;
  border-radius: 999px !important;
  padding: 3px 10px !important;
}

.pulse-dot-green {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  display: inline-block;
  background: #10b981;
  box-shadow: 0 0 6px #10b981;
}

.terminal-screen {
  padding: 8px !important;
}

.terminal-scroller {
  max-height: 560px !important;
  overflow-y: auto !important;
  display: flex !important;
  flex-direction: column-reverse !important;
  padding: 6px !important;
}

.tr-row {
  display: grid !important;
  grid-template-columns: 70px 130px 110px 1fr !important;
  gap: 10px !important;
  padding: 7px 8px !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
  font-size: 0.82rem !important;
  line-height: 1.45 !important;
  color: #cbd5e1 !important;
}

.tr-row:hover {
  background: rgba(255, 255, 255, 0.02) !important;
}

.tr-ts {
  opacity: 0.45 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.74rem !important;
}

.tr-ag {
  font-weight: 600 !important;
  color: #f8fafc !important;
}

.tr-kd {
  display: inline-block !important;
  text-align: center !important;
  padding: 1px 8px !important;
  border-radius: 5px !important;
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  font-family: 'JetBrains Mono', monospace !important;
}

.kd-THOUGHT { background: rgba(168, 85, 247, 0.2) !important; color: #c084fc !important; border: 1px solid rgba(168, 85, 247, 0.4) !important; }
.kd-ACTION { background: rgba(56, 189, 248, 0.2) !important; color: #38bdf8 !important; border: 1px solid rgba(56, 189, 248, 0.4) !important; }
.kd-OBSERVATION { background: rgba(52, 211, 153, 0.2) !important; color: #34d399 !important; border: 1px solid rgba(52, 211, 153, 0.4) !important; }
.kd-DECISION { background: rgba(251, 191, 36, 0.2) !important; color: #fbbf24 !important; border: 1px solid rgba(251, 191, 36, 0.4) !important; }
.kd-MESSAGE { background: rgba(244, 114, 182, 0.2) !important; color: #f472b6 !important; border: 1px solid rgba(244, 114, 182, 0.4) !important; }
.kd-STATE { background: rgba(148, 163, 184, 0.2) !important; color: #94a3b8 !important; border: 1px solid rgba(148, 163, 184, 0.4) !important; }

.tr-tx {
  word-break: break-word !important;
}

@media(max-width: 992px) {
  .snapshot-charts-row { grid-template-columns: 1fr !important; }
}
"""

THEME = gr.themes.Base(  # type: ignore
    primary_hue="cyan",  # type: ignore
    secondary_hue="amber",  # type: ignore
    neutral_hue="slate",  # type: ignore
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Outfit"), "system-ui", "sans-serif"],  # type: ignore
).set(
    body_background_fill="#0b0f19",
    body_background_fill_dark="#0b0f19",
    body_text_color="#f8fafc",
    body_text_color_dark="#f8fafc",
    block_background_fill="#121826",
    block_background_fill_dark="#121826",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_color_dark="rgba(255, 255, 255, 0.08)",
    block_border_width="1px",
    block_radius="16px",
    input_background_fill="#141c2e",
    input_background_fill_dark="#141c2e",
    input_border_color="#243048",
    input_border_color_dark="#243048",
    input_border_width="1px",
    input_radius="8px",
)

# Default configuration for B.Tech 2nd Year Term 1 subjects (optional toggle, confidence, quiz score, exam weight)
DEMO_DEFAULTS = {
    "Probability and Statistics": (True, 4, 65, 5),
    "DSA C++": (True, 5, 78, 5),
    "Fundamentals of Artificial Intelligence": (True, 3, 52, 4),
    "Advanced DBMS": (True, 4, 70, 4),
}


def _need_plan(o: Optional[Orchestrator]) -> bool:
    return o is None or o.state.plan is None


import calendar

def render_mini_calendar(exam_str: str = "Nov 01, 2026") -> str:
    today = date.today()
    parsed_exam = None
    parse_error = False
    try:
        parsed_exam = parse_exam_date(str(exam_str).strip())
        days_left = (parsed_exam - today).days
    except Exception:
        parse_error = True
        days_left = 30

    month_name = today.strftime("%B %Y")
    cal = calendar.monthcalendar(today.year, today.month)

    days_html = []
    for week in cal:
        for day in week:
            if day == 0:
                days_html.append("<span></span>")
            else:
                is_today = (day == today.day)
                is_exam = (parsed_exam is not None and parsed_exam.year == today.year and parsed_exam.month == today.month and parsed_exam.day == day)
                if is_today and is_exam:
                    days_html.append(f"<span class='active-cyan' title='Today & Exam ({today.strftime('%b %d')})'>{day}</span>")
                elif is_today:
                    days_html.append(f"<span class='active-cyan' title='Today ({today.strftime('%b %d')})'>{day}</span>")
                elif is_exam and parsed_exam is not None:
                    days_html.append(f"<span class='active-blue' title='Target Exam ({parsed_exam.strftime('%b %d')})'>{day}</span>")
                else:
                    days_html.append(f"<span>{day}</span>")

    # Countdown banner
    if parse_error or parsed_exam is None:
        countdown_html = f"""
        <div class='live-countdown-banner'>
          <div style='display:flex; flex-direction:column; gap:2px;'>
            <span style='font-weight:600; color:#f87171;'>Target Exam Date</span>
            <span style='font-size:0.72rem; color:#94a3b8;'>Enter format: Nov 01, 2026 or YYYY-MM-DD</span>
          </div>
          <span class='days-count' style='background:#ef4444;'>Invalid</span>
        </div>
        """
        exam_badge = ""
    elif days_left > 0:
        assert parsed_exam is not None
        countdown_html = f"""
        <div class='live-countdown-banner'>
          <div style='display:flex; flex-direction:column; gap:2px;'>
            <span style='font-weight:600; color:#e2e8f0;'>Live Exam Countdown</span>
            <span style='font-size:0.72rem; color:#94a3b8;'>Counting from Today: {today.strftime('%b %d, %Y')} &rarr; Exam: {parsed_exam.strftime('%b %d, %Y')}</span>
          </div>
          <span class='days-count'>{days_left} Days Left</span>
        </div>
        """
        if parsed_exam.year != today.year or parsed_exam.month != today.month:
            exam_badge = f"""
            <div style='font-size:0.75rem; color:#94a3b8; margin-top:-12px; margin-bottom:14px; text-align:center;'>
              Target Exam: <span style='color:#38bdf8; font-weight:700;'>{parsed_exam.strftime('%b %d, %Y')}</span> ({days_left} days away)
            </div>
            """
        else:
            exam_badge = ""
    elif days_left == 0:
        countdown_html = f"""
        <div class='live-countdown-banner'>
          <div style='display:flex; flex-direction:column; gap:2px;'>
            <span style='font-weight:600; color:#38bdf8;'>Target Exam is TODAY!</span>
            <span style='font-size:0.72rem; color:#94a3b8;'>Good luck on your examination!</span>
          </div>
          <span class='days-count' style='background:#10b981;'>Exam Day</span>
        </div>
        """
        exam_badge = ""
    else:
        assert parsed_exam is not None
        countdown_html = f"""
        <div class='live-countdown-banner'>
          <div style='display:flex; flex-direction:column; gap:2px;'>
            <span style='font-weight:600; color:#f59e0b;'>Exam Date in Past</span>
            <span style='font-size:0.72rem; color:#94a3b8;'>Was on {parsed_exam.strftime('%b %d, %Y')}</span>
          </div>
          <span class='days-count' style='background:#f59e0b;'>{abs(days_left)}d ago</span>
        </div>
        """
        exam_badge = ""

    return f"""
    <div class='schedule-card'>
      <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
        <div class='schedule-header' style='margin-bottom:0;'>Schedule Overview</div>
        <div style='font-size:0.76rem; font-weight:700; color:#38bdf8; background:rgba(56,189,248,0.12); padding:3px 8px; border-radius:6px; border:1px solid rgba(56,189,248,0.25);'>{month_name}</div>
      </div>

      {countdown_html}

      <!-- Mini Month Calendar -->
      <div class='mini-cal-weekdays'>
        <span>M</span><span>T</span><span>W</span><span>T</span><span>F</span><span>S</span><span>S</span>
      </div>
      <div class='mini-cal-days'>
        {''.join(days_html)}
      </div>
      {exam_badge}

      <!-- Timeline Blocks -->
      <div class='timeline-blocks-list'>
        <div class='timeline-block-item'>
          <span class='block-time'>10:00</span>
          <div class='block-pill block-blue'>
            <span>Study block 1</span>
            <span class='block-sub'>DSA C++ Practice</span>
          </div>
        </div>
        <div class='timeline-block-item'>
          <span class='block-time'>09:00</span>
          <div class='block-pill block-copper'>
            <span>Study block 2</span>
            <span class='block-sub'>Probability &amp; Stats</span>
          </div>
        </div>
        <div class='timeline-block-item'>
          <span class='block-time'>05:00</span>
          <div class='block-pill block-copper'>
            <span>Study block 3</span>
            <span class='block-sub'>AI Heuristic Search</span>
          </div>
        </div>
        <div class='timeline-block-item'>
          <span class='block-time'>03:00</span>
          <div class='block-pill block-blue'>
            <span>Study block 4</span>
            <span class='block-sub'>DBMS Normalization</span>
          </div>
        </div>
      </div>
    </div>
    """

def build_plan(orch, name, exam_str, target, hours, start_hour, *rows):
    try:
        LLM.refresh()
        exam = parse_exam_date(str(exam_str).strip())
        days = (exam - date.today()).days
        if days < 2 or days > MAX_HORIZON_DAYS:
            raise ValueError(f"Exam date must be between 2 and {MAX_HORIZON_DAYS} days from today (you entered {days} days from today).")
        subs: List[SubjectInput] = []
        for i, sname in enumerate(DEMO_DEFAULTS.keys()):
            inc, conf, score, wt, chosen_mods = rows[i * 5: (i + 1) * 5]
            if inc:
                selected = list(chosen_mods) if chosen_mods else []
                if not selected:
                    raise ValueError(f"No modules selected for '{sname}'. Please select at least one module or uncheck the subject.")
                subs.append(SubjectInput(name=sname, confidence=int(conf), quiz_score=float(score), exam_weight=int(wt), selected_modules=selected))
        if not subs:
            raise ValueError("Select at least one subject to study.")
        prof = StudentProfile(name=(name or "Alex").strip(), exam_date=exam, target_score=int(target),
                               hours_per_day=float(hours), start_hour=int(start_hour), subjects=subs)
        o = Orchestrator()
        o.onboard(prof)
        mode = f"OpenAI {LLM.MODEL}" if LLM.live else "offline rule engine"
        msg = f"**Plan v1 is ready** for {len(o.state.topics)} topics across {len(subs)} subjects over {days} days (engine: {mode}). Open the **Schedule & Roadmap** tab."
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


with gr.Blocks(theme=THEME, css=CSS, title="studyplanner.ai/dashboard") as demo:
    orch_state = gr.State(None)

    # Hero Banner exactly matching Image with Title, 3 Crests & 4 Badges (Zero Watermarks)
    gr.HTML("""
    <div class='hero-image2'>
      <div class='hero-title-group'>
        <h1>Personalized Academic Mastery</h1>
        <p>A Sophisticated AI Study &amp; Performance Agent</p>
      </div>
      <div class='hero-crests-row'>
        <!-- Left Blue Rosette -->
        <svg width="44" height="54" viewBox="0 0 50 64" fill="none">
          <circle cx="25" cy="22" r="16" fill="rgba(56,189,248,0.15)" stroke="#38bdf8" stroke-width="2"/>
          <path d="M18 22L23 27L32 17" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M19 36L14 54L25 47L36 54L31 36" fill="rgba(56,189,248,0.25)" stroke="#38bdf8" stroke-width="1.5"/>
        </svg>

        <!-- Center Master Emblem: Laurel + Lightning + Books -->
        <svg width="74" height="74" viewBox="0 0 100 100" fill="none">
          <!-- Left laurel (cyan) -->
          <path d="M28 72C22 62 20 48 24 35C25 32 28 34 27 37C24 48 26 58 31 66C32 68 30 71 28 72Z" fill="#38bdf8"/>
          <path d="M23 42C17 40 14 34 16 28C18 34 23 37 25 38C26 39 25 41 23 42Z" fill="#38bdf8"/>
          <path d="M20 54C14 53 12 47 14 41C16 47 21 49 23 50C23 52 22 53 20 54Z" fill="#38bdf8"/>
          <path d="M21 66C16 65 14 60 16 54C18 59 23 61 24 62C24 64 23 65 21 66Z" fill="#38bdf8"/>
          <path d="M27 30C23 27 22 21 26 16C26 22 30 25 32 26C31 28 29 29 27 30Z" fill="#38bdf8"/>
          <path d="M35 22C32 18 33 12 38 8C37 14 40 18 41 20C40 21 37 22 35 22Z" fill="#38bdf8"/>

          <!-- Right laurel (copper) -->
          <path d="M72 72C78 62 80 48 76 35C75 32 72 34 73 37C76 48 74 58 69 66C68 68 70 71 72 72Z" fill="#fb923c"/>
          <path d="M77 42C83 40 86 34 84 28C82 34 77 37 75 38C74 39 75 41 77 42Z" fill="#fb923c"/>
          <path d="M80 54C86 53 88 47 86 41C84 47 79 49 77 50C77 52 78 53 80 54Z" fill="#fb923c"/>
          <path d="M79 66C84 65 86 60 84 54C82 59 77 61 76 62C76 64 77 65 79 66Z" fill="#fb923c"/>
          <path d="M73 30C77 27 78 21 74 16C74 22 70 25 68 26C69 28 71 29 73 30Z" fill="#fb923c"/>
          <path d="M65 22C68 18 67 12 62 8C63 14 60 18 59 20C60 21 63 22 65 22Z" fill="#fb923c"/>

          <!-- Central Lightning Bolt -->
          <path d="M52 14L37 42H50L45 62L65 34H51L56 14H52Z" fill="#e0f2fe" filter="drop-shadow(0 0 6px rgba(56,189,248,0.5))"/>

          <!-- Stacked Books Beneath -->
          <path d="M38 68L49 64L62 68L51 72L38 68Z" fill="#e0f2fe"/>
          <path d="M36 74L49 70L64 74L51 78L36 74Z" fill="#38bdf8"/>
          <path d="M34 80L49 76L66 80L51 84L34 80Z" fill="#fb923c"/>
        </svg>

        <!-- Right Pink Rosette -->
        <svg width="44" height="54" viewBox="0 0 50 64" fill="none">
          <circle cx="25" cy="22" r="16" fill="rgba(244,114,182,0.15)" stroke="#f472b6" stroke-width="2"/>
          <path d="M18 22L23 27L32 17" stroke="#f472b6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M19 36L14 54L25 47L36 54L31 36" fill="rgba(244,114,182,0.25)" stroke="#f472b6" stroke-width="1.5"/>
        </svg>
      </div>
      <div class='hero-chips-row'>
        <span class='hero-pill-badge'>👥 3 Cooperating Agents</span>
        <span class='hero-pill-badge'><span class='dot-pink'></span> Bayesian Knowledge Tracing</span>
        <span class='hero-pill-badge'><span class='icon-copper'>⚡</span> Spaced Repetition SM-2</span>
        <span class='hero-pill-badge'><span class='dot-purple'></span> Dynamic Roadmap Optimization</span>
      </div>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("⚙️ Setup & Diagnostics"):
            # BALANCED TWO-COLUMN DASHBOARD LAYOUT (Matching Reference Design)
            with gr.Row(elem_classes=["dash-main-container"]):
                # ==========================================
                # LEFT COLUMN: Portfolio & Plan Diagnostics (~28% width)
                # ==========================================
                with gr.Column(scale=3, min_width=290, elem_classes=["left-diag-column"]):
                    with gr.Group(elem_classes=["diagnostics-card"]):
                        gr.HTML("""
                        <div class='card-header-bar'>
                            <div class='card-header-title'>Portfolio &amp; Plan Diagnostics</div>
                            <svg width='90' height='26' viewBox='0 0 100 26' fill='none'>
                                <path d='M0 20C20 20 35 10 55 14C75 18 85 4 100 2' stroke='#38bdf8' stroke-width='2.5' stroke-linecap='round'/>
                            </svg>
                        </div>
                        """)
                        s_name = gr.Textbox(label="Student Profile", value="Alex", elem_classes=["profile-input-box"])
                        s_exam = gr.Textbox(label="Target Exam", value="Nov 01, 2026", elem_classes=["exam-input-box"], placeholder="Nov 01, 2026 or 2026-11-01")
                        s_target = gr.Slider(50, 100, value=85, step=1, label="Target Mastery Score (%)")
                        s_hours = gr.Slider(0.75, 10, value=3, step=0.25, label="Daily Study Capacity (Hours)")
                        s_start = gr.Slider(5, 21, value=17, step=1, label="Daily Study Start Hour (24h clock)")
                        build_btn = gr.Button("🚀 Diagnose & Build My Plan", variant="primary", size="lg", elem_classes=["build-plan-button"])
                        build_status = gr.Markdown()

                # ==========================================
                # RIGHT COLUMN: Main Setup & Diagnostics Area (~72% width)
                # ==========================================
                with gr.Column(scale=7, elem_classes=["right-main-column"]):
                    gr.HTML("""
                    <div class='dash-main-title'>Portfolio &amp; Plan Setup and Diagnostics</div>
                    """)

                    # TOP ROW: Subject Portfolio (Left) + Schedule Overview (Right)
                    with gr.Row(elem_classes=["top-cards-row"]):
                        # Top-Left: Subject Portfolio
                        with gr.Column(scale=11, elem_classes=["subject-portfolio-col"]):
                            with gr.Group(elem_classes=["portfolio-group-card"]):
                                gr.HTML("""
                                <div class='card-header-bar'>
                                    <div class='card-header-title'>Subject Portfolio <span class='info-icon' title='Select subjects, customize confidence, quiz score and target modules'>ⓘ</span></div>
                                    <svg width='90' height='26' viewBox='0 0 100 26' fill='none'>
                                        <path d='M0 18C20 18 40 6 60 12C80 18 88 4 100 2' stroke='#fb923c' stroke-width='2.5' stroke-linecap='round'/>
                                        <circle cx='88' cy='4' r='3.5' fill='#fb923c' stroke='#131b2e' stroke-width='1.5'/>
                                    </svg>
                                </div>
                                """)
                                subject_inputs: List[Any] = []
                                for sname, (inc, conf, score, wt) in DEMO_DEFAULTS.items():
                                    with gr.Group(elem_classes=["subject-item-box"]):
                                        with gr.Row():
                                            c_inc = gr.Checkbox(value=inc, label=f"{SUBJECT_ICONS.get(sname, '📚')} {sname}", scale=3)
                                            c_wt = gr.Slider(1, 5, value=wt, step=1, label="Weight (1-5)", scale=2)
                                        with gr.Row():
                                            c_conf = gr.Slider(1, 5, value=conf, step=1, label="Confidence (1-5)", scale=1)
                                            c_score = gr.Slider(0, 100, value=score, step=1, label="Last Quiz (%)", scale=1)
                                        with gr.Accordion(f"📂 Select Modules for {sname} ({len(CURRICULUM[sname])} Available)", open=False, elem_classes=["module-accordion"]):
                                            c_mods = gr.CheckboxGroup(
                                                choices=list(CURRICULUM[sname].keys()),
                                                value=list(CURRICULUM[sname].keys()),
                                                label=f"Pick modules for {sname}:",
                                                elem_classes=["module-checkboxes"]
                                            )
                                    subject_inputs += [c_inc, c_conf, c_score, c_wt, c_mods]

                        # Top-Right: Schedule Overview Live Calendar
                        with gr.Column(scale=9, elem_classes=["schedule-overview-col"]):
                            schedule_overview_html = gr.HTML(render_mini_calendar("Nov 01, 2026"))

                    # BOTTOM ROW: Performance Snapshot (spans full width under Subject Portfolio & Schedule Overview)
                    with gr.Row(elem_classes=["snapshot-row"]):
                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class='snapshot-card'>
                              <div class='snapshot-header'>
                                <div class='snapshot-title'>Performance Snapshot</div>
                                <div class='snapshot-legend'>
                                  <span><span class='legend-line-blue'>&mdash;</span> Domains</span>
                                  <span><span class='legend-line-pink'>&mdash;</span> DSA C++</span>
                                </div>
                              </div>
                              <div class='snapshot-charts-row'>
                                <!-- Left: Smooth Blue Area Curve -->
                                <div class='chart-col'>
                                  <svg width="100%" height="90" viewBox="0 0 130 90" fill="none">
                                    <defs>
                                      <linearGradient id="areaGradBlue" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
                                        <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
                                      </linearGradient>
                                    </defs>
                                    <path d="M0 75C20 75 35 60 55 25C75 0 95 65 110 50C120 40 125 45 130 45L130 90L0 90Z" fill="url(#areaGradBlue)"/>
                                    <path d="M0 75C20 75 35 60 55 25C75 0 95 65 110 50C120 40 125 45 130 45" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/>
                                  </svg>
                                </div>

                                <!-- Center: Multi-line chart (Pink & Copper curves with axes) -->
                                <div class='chart-col'>
                                  <svg width="100%" height="90" viewBox="0 0 180 90" fill="none">
                                    <defs>
                                      <linearGradient id="gradPink" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stop-color="#f472b6" stop-opacity="0.3"/>
                                        <stop offset="100%" stop-color="#f472b6" stop-opacity="0.0"/>
                                      </linearGradient>
                                    </defs>
                                    <!-- Horizontal gridlines -->
                                    <line x1="20" y1="15" x2="175" y2="15" stroke="rgba(255,255,255,0.06)"/>
                                    <line x1="20" y1="45" x2="175" y2="45" stroke="rgba(255,255,255,0.06)"/>
                                    <line x1="20" y1="75" x2="175" y2="75" stroke="rgba(255,255,255,0.08)"/>
                                    <!-- Y-axis text -->
                                    <text x="5" y="18" fill="#64748b" font-size="8">40</text>
                                    <text x="5" y="48" fill="#64748b" font-size="8">20</text>
                                    <text x="5" y="78" fill="#64748b" font-size="8">0</text>
                                    <!-- Curves -->
                                    <path d="M25 75C55 70 70 65 95 50C120 35 140 38 170 18L170 75Z" fill="url(#gradPink)"/>
                                    <path d="M25 75C55 70 70 65 95 50C120 35 140 38 170 18" stroke="#f472b6" stroke-width="2" stroke-linecap="round"/>
                                    <path d="M25 72C50 68 75 75 105 52C130 35 150 48 170 28" stroke="#fb923c" stroke-width="2" stroke-linecap="round"/>
                                    <!-- X-axis labels -->
                                    <text x="25" y="87" fill="#64748b" font-size="7">Jan</text>
                                    <text x="60" y="87" fill="#64748b" font-size="7">Feb</text>
                                    <text x="95" y="87" fill="#64748b" font-size="7">Mar</text>
                                    <text x="130" y="87" fill="#64748b" font-size="7">Apr</text>
                                    <text x="160" y="87" fill="#64748b" font-size="7">May</text>
                                  </svg>
                                </div>

                                <!-- Right: Mini Bar Chart (Blue & Pink columns) -->
                                <div class='chart-col'>
                                  <svg width="100%" height="90" viewBox="0 0 120 90" fill="none">
                                    <line x1="5" y1="75" x2="115" y2="75" stroke="rgba(255,255,255,0.08)"/>
                                    <!-- Bars -->
                                    <rect x="15" y="52" width="10" height="23" rx="2" fill="#38bdf8"/>
                                    <rect x="30" y="46" width="10" height="29" rx="2" fill="#f472b6"/>
                                    <rect x="52" y="32" width="10" height="43" rx="2" fill="#38bdf8"/>
                                    <rect x="67" y="24" width="10" height="51" rx="2" fill="#38bdf8"/>
                                    <rect x="88" y="20" width="10" height="55" rx="2" fill="#38bdf8"/>
                                    <rect x="103" y="36" width="10" height="39" rx="2" fill="#f472b6"/>
                                    <!-- Labels -->
                                    <text x="16" y="86" fill="#64748b" font-size="7">P&amp;S</text>
                                    <text x="54" y="86" fill="#64748b" font-size="7">DSA</text>
                                    <text x="76" y="86" fill="#64748b" font-size="7">AI</text>
                                    <text x="104" y="86" fill="#64748b" font-size="7">DBMS</text>
                                  </svg>
                                </div>
                              </div>
                            </div>
                            """)

            # Diagnostic Output Block when Plan is Synthesized (Bento Cards + Coach Brief + Priority Need Table)
            diag_html = gr.HTML(EMPTY)

        with gr.Tab("📅 Schedule & Roadmap"):
            summary_html = gr.HTML(EMPTY)
            with gr.Accordion("⚡ Life Happened? Rebalance the Plan", open=True):
                with gr.Row():
                    a_missed = gr.Slider(0, 14, value=0, step=1, label="Days Missed / Lost")
                    a_fatigue = gr.Slider(1, 5, value=2, step=1, label="Current Fatigue Level (1 Fresh &rarr; 5 Exhausted)")
                    a_hours = gr.Slider(0, 10, value=0, step=0.25, label="Updated Daily Hours (0 retains current capacity)")
                a_note = gr.Textbox(label="Context / Circumstance (Optional)", placeholder="e.g. college lab exams, caught a fever, or need more review")
                with gr.Row():
                    adapt_btn = gr.Button("⚡ Adapt & Rebalance Schedule", variant="primary")
                    done_btn = gr.Button("✅ Complete Today's Plan & Advance Cursor")
                adapt_msg = gr.Markdown()
            timeline_html = gr.HTML(EMPTY)
            horizon = gr.Slider(7, 30, value=14, step=1, label="Timeline Horizon (Days shown in detailed breakdown)")
            table_md = gr.Markdown("_No study plan generated yet. Synthesize your plan in the first tab._")

        with gr.Tab("⚡ Quiz Arena"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        q_n = gr.Slider(3, 6, value=4, step=1, label="Number of Questions")
                        q_btn = gr.Button("🎯 Generate Targeted Quiz on Weakest Topics", variant="primary")
                    quiz_md = gr.Markdown("_Generate a quiz to begin assessment._")
                    ans_boxes = [gr.Textbox(lines=3, visible=False, label=f"Your answer to Q{i + 1}") for i in range(6)]
                    submit_btn = gr.Button("📤 Submit Answers for Cognitive Grading", variant="primary")
                    feedback_md = gr.Markdown()
                with gr.Column(scale=2):
                    analytics_html = gr.HTML(EMPTY)

        with gr.Tab("🟣 Agent Telemetry"):
            gr.Markdown("### 🔍 Transparent Multi-Agent Cognitive Trace\nInspect every thought, action, tool invocation, observation, decision, and inter-agent message in real time.")
            trace_html = gr.HTML(render_trace(None))
            trace_btn = gr.Button("🔄 Refresh Telemetry Audit Stream")

    VIEW = [timeline_html, table_md, summary_html, analytics_html, trace_html]
    build_btn.click(build_plan, [orch_state, s_name, s_exam, s_target, s_hours, s_start, *subject_inputs],
                    [orch_state, build_status, diag_html, *VIEW])
    s_exam.change(render_mini_calendar, [s_exam], [schedule_overview_html])
    adapt_btn.click(do_adapt, [orch_state, a_missed, a_fatigue, a_hours, a_note], [orch_state, adapt_msg, *VIEW])
    done_btn.click(do_complete, [orch_state], [orch_state, adapt_msg, *VIEW])
    horizon.change(set_horizon, [orch_state, horizon], [orch_state, table_md])
    q_btn.click(make_quiz, [orch_state, q_n], [orch_state, quiz_md, *ans_boxes, feedback_md, *VIEW])
    submit_btn.click(submit_quiz, [orch_state, *ans_boxes], [orch_state, feedback_md, *VIEW])
    trace_btn.click(refresh_trace, [orch_state], [trace_html])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
