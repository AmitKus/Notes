# README

## LLamaIndex

- (Complex Query Resolution through LlamaIndex Utilizing Recursive Retrieval, Document Agents, and Sub Question Query Decomposition)[https://medium.com/@sauravjoshi23/complex-query-resolution-through-llamaindex-utilizing-recursive-retrieval-document-agents-and-sub-d4861ecd54e6]


## DSPY: COMPILING DECLARATIVE LANGUAGE MODEL CALLS INTO SELF-IMPROVING PIPELINES

- This work proposes the first programming model that translates prompting techniques into parameterized declarative modules and introduces an effective compiler with general optimization strategies (teleprompters) to optimize arbitrary pipelines of these modules.
### DSPy program
- DSPy programs are expressed in Python: each program takes the task input (e.g., a question to answer or a paper to summarize) and returns the output (e.g., an answer or a summary) after a series of steps. 
- DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters.
	- Signatures abstract the input/output behavior of a module; 
	- modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and 
	- teleprompters optimize all modules in the pipeline to maximize a metric.
#### Notes
1. NATURAL LANGUAGE SIGNATURES CAN ABSTRACT PROMPTING & FINETUNING
2. PARAMETERIZED & TEMPLATED MODULES CAN ABSTRACT PROMPTING TECHNIQUES
3. TELEPROMPTERS CAN AUTOMATE PROMPTING FOR ARBITRARY PIPELINES
### DSPY Compiler

1. Stage 1 Candidate generation
	- While LMs can be highly unreliable, we find they can be rather efficient at searching the space of solutions for multi-stage designs.
2. Stage 2 Optimization
	- Now each parameter has a discrete set of candidates: demonstrations, instructions, etc. Many hyperparameter tuning algorithms (e.g., random search or Treestructured Parzen Estimators as in HyperOpt (Bergstra et al., 2013) and Optuna (Akiba et al., 2019)) can be applied for selection among candidates.
3. Stage 3 Higher-order program optimization
	- A different type of optimization that the DSPy compiler supports is modifying the control flow of the program. One of the simplest forms of these is ensembles,

### Goals of evaluation
- H1 With DSPy, we can replace hand-crafted prompt strings with concise and well-defined modules, without reducing quality or expressive power. 
- H2 Parameterizing the modules and treating prompting as an optimization problem makes DSPy better at adapting to different LMs, and it may outperform expert-written prompts. 
- H3 The resulting modularity makes it possible to more thoroughly explore complex pipelines that have useful performance characteristics or that fit nuanced metrics.