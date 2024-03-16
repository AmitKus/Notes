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