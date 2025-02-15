
### Core question
what is the simplest approach to achieve both test-time scaling and strong reasoning performance?

![](attachments/39c8d606b1fdb86cdbd6b9c3e9eab7a6_MD5.jpeg)
### Procedure

1. If the model generates more thinking tokens than a desired limit, we forcefully end the thinking process by appending an end-of-thinking token delimiter. Ending the thinking this way makes the model transition to generating its answer. 
2. If we want the model to spend more test-time compute on a problem, we suppress the generation of the end-of-thinking token delimiter and instead append “Wait”
