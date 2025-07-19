# Notes from practice exams










- `Knowledge Bases for Amazon Bedrock` takes care of the entire ingestion workflow of converting your documents into embeddings (vector) and storing the embeddings in a specialized vector database. `Knowledge Bases for Amazon Bedrock `supports popular databases for vector storage, including vector engine for Amazon OpenSearch Serverless, Pinecone, Redis Enterprise Cloud, Amazon Aurora, and MongoDB.
  - If you do not have an existing vector database, `Amazon Bedrock creates an OpenSearch Serverless vector` store for you.













## ML 
- `Hyperparameter tuning` is a method to adjust the behavior of an ML algorithm. You can make changes to an ML model by using hyperparameter tuning to modify the behavior of the algorithm.
- `Feature engineering` is a method to select and transform variables when you create a predictive model. Feature engineering includes feature creation, feature transformation, feature extraction, and feature selection. Feature engineering enhances the data by increasing the number of variables in the training dataset to ultimately improve model performance.
- `Model evaluation` is a step in the ML development pipeline that occurs after model training. You can use model evaluation to evaluate a model's performance and metrics. Model evaluation does not increase the number of variables in the training dataset or modify the behavior of the algorithm.
- `Model monitoring` is a component of the ML lifecycle that captures data and compares the data to the training data. You can use model monitoring to identify data quality issues, model quality issues, bias drift, and feature attribution drift. Model monitoring does not increase the number of variables in the training dataset or modify the behavior of the algorithm.
- `Data collection` is a step to label, ingest, and aggregate data that you will use for ML model training. During data collection, you ingest and aggregate data from multiple sources. Then, you label the data. The data collection stage does not increase the number of variables in the training dataset or modify the behavior of the algorithm.

- `Foundation Model` are large models that are pre-trained on a vast amount of data and that can perform several tasks. FMs can be fine-tuned for downstream tasks by using smaller datasets.


- `Underfitting` occurs when a model does not identify the relationships in the training data. Underfitting would lead to low accuracy on both the training data and the testing data.
- `Overfitting` is when a model learns from the training data but is unable to perform well when given new data. Overfitting explains why the model has high accuracy on the training data but has low accuracy on the testing data

- `Real-time inference` is suitable for use cases with low latency or high throughput requirements. Real-time inference supports processing times of 60 seconds. Real-time inference provides a persistent and fully managed endpoint to handle traffic. Real-time inference offers the lowest latency requirements because of the 60-second processing times.
- `Asynchronous inference` is suitable for use cases with larger datasets and processing times of up to 1 hour. Asynchronous inference can queue incoming requests for inference processing. Asynchronous inference provides moderate latency requirements because of the processing times of up to 1 hour.
- `Batch transform` is suitable for offline processing when data can be processed in batches. Batch transform can support processing times of days. Therefore, batch transform provides the highest latency requirements of these options.

### Training Types
- `Fine-tuning` is the process to further train and refine a pre-trained LLM on a smaller, targeted dataset. The purpose of fine-tuning a pre-trained LLM is to maintain the original capability of the model and adapt to more specialized use cases. Fine-tuning requires additional development effort to train the model.
- `RAG` is the process of improving the quality and consistency of LLMs by referencing an external knowledge base that is outside of the LLM's training data sources. RAG references the external knowledge base before generating a response. You can use RAG to provide the model with access to external sources of knowledge with minimal development effort.
- `In-context learning` is the process of providing a few examples to help an LLM better align responses to an expected format or output. In-context learning is also referred to as `few-shot prompting`. In-context learning does not increase the consistency and quality of an LLM by providing the model with access to external sources of knowledge.

### Prompting
- `Prompt engineering` is the process of designing and refining the input prompts for an LLM to generate a specific type of output. Prompt engineering involves selecting appropriate keywords and shaping the input so that the model can produce your desired outcomes. Prompt engineering does not increase the consistency and quality of an LLM by providing the model with access to external sources of knowledge.
-  `Zero-shot prompting` without examples is less effective for tasks that require a specific writing style or format. Zero-shot prompting without examples might cause the model to struggle to infer the desired output.
  - without examples is less effective for tasks that require a specific writing style or format. Zero-shot prompting without examples might cause the model to struggle to infer the desired output.



| Prompting Type   | Definition                                                                                                                                              | Examples Provided | Reliance On                                                    | Task Complexity                                       | Performance                                                       | Use Cases                                                                                                |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------- | :------------------------------------------------------------- | :---------------------------------------------------- | :---------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| **Zero-Shot** | No examples are given in the prompt. The model relies solely on its pre-existing knowledge from training to understand the instruction and generate a response. | None              | Model's pre-trained knowledge                                  | Simple, common, or well-understood tasks              | Varies; can be less reliable for novel tasks or specific formats  | Basic sentiment analysis, straightforward Q&A, simple summarization                                      |
| **One-Shot** | A single example of an input-output pair is provided in the prompt to clarify the expected format, style, or specific understanding of the task.           | One               | Pre-trained knowledge + single example for context/format      | Slightly more specific tasks, or clarifying output format | Generally better than zero-shot; provides a hint                  | Simple classification, basic rephrasing, formatting guidance                                             |
| **Few-Shot** | A small number (typically 2-5 or more) of input-output examples are provided in the prompt, allowing the model to identify patterns and desired behavior. | Few (2-5+)        | Pre-trained knowledge + multiple examples for pattern recognition | More complex, nuanced, or custom tasks needing specific patterns or styles | Often significantly improves performance and accuracy, especially for specific tasks | Complex classification, data extraction, creative writing (style transfer), code generation, complex translations |



### Soring and Evaluation
- `Recall-Oriented Understudy for Gisting Evaluation (ROUGE)` is a metric that you can use to evaluate the quality of text summarization and text generation. You can use ROUGE to assess the performance of an FM for text generation.
- `F1 score` to evaluate a model's accuracy for binary classification. F1 scores use precision and recall to evaluate how accurate a model correctly classifies the correct class. You cannot use the `F1 score` to assess the performance of an FM for text generation



## AWS Services

### AI
- `Amazon Bedrock` is a fully managed service that provides a unified API to access popular foundation models (FMs). Amazon Bedrock supports image generation models from providers such as Stability AI or AWS. You can use Amazon Bedrock to consume FMs through a unified API without the need to train, host, or manage ML models. This is the most suitable solution for a company that does not want to train or manage ML models for image generation.
- `SageMaker JumpStart` is a feature of SageMaker that includes pre-trained foundation models (FMs) for image generation. You can host the models in SageMaker with no additional training. However, this solution requires you to configure and monitor the production endpoint that hosts the ML model.
  - `SageMaker Model Cards` to create records and to document details about ML models in a single place. SageMaker Model Cards support transparent and explainable model development by providing comprehensive, immutable documentation of essential model information.
  - `SageMaker Role Manager` to define user permissions for ML activities. You cannot use SageMaker Role Manager to create a record of essential model information.
  - `SageMaker Model Dashboard` is a central place to view, search, and explore all models in an AWS account. SageMaker Model Dashboard provides insights into model deployment, usage, performance tracking, and monitoring. You cannot use SageMaker Model Dashboard to create a record of essential model information, such as risk ratings, training details, and evaluation results.
  - `SageMaker Model Monitor` monitors the quality of ML models and data in production. You cannot use SageMaker Model Monitor to create a record of essential model information such as risk ratings, training details, and evaluation results.
- `Amazon Rekognition` is a fully managed AI service that uses deep learning to analyze images and videos. Amazon Rekognition can perform object-detection tasks. However, Amazon Rekognition does not modify or generate new images.
  - Can provide references images to the model to better train it
- `Amazon Personalize` is a fully managed ML service that targets recommendations, such as search results or user segments based on interaction data. You can use Amazon Personalize to target a marketing campaign. For example, Amazon Personalize can recommend segments of users who are most likely to respond to a promotion. However, Amazon Personalize is not an image generation service.
-  `Amazon Textract` is a service that you can use to extract text and data from scanned documents, PDFs, and images. You cannot use Amazon Textract to identify new product categories based on historic images.
- `Amazon Polly` is a text-to-speech (TTS) service that can convert text into lifelike speech. You cannot use Amazon Polly to detect and extract text, handwriting, and data from invoice images.
- `Amazon Kendra` is an intelligent search service that uses semantic and contextual understanding to provide relevant responses to a search query. You cannot use Amazon Kendra to detect and extract text, handwriting, and data from invoice images.
- `Amazon Comprehend` is a natural language processing (NLP) service that can extract insights and relationships from text data. You cannot use Amazon Comprehend to process textual information from images that are provided in PNG format. Amazon Comprehend requires text as input.
- `Amazon Transcribe` is a service that you can use to convert speech into text. You can use Amazon Transcribe to facilitate the transcription of audio recordings. If media contains domain-specific or non-standard terms, you can use a custom vocabulary or a custom model to improve the accuracy of the transcriptions. Examples of domain-specific or non-standard terms include brand names, acronyms, technical words, and jargon. A solution that uses a custom language model in Amazon Transcribe can improve transcription accuracy for domain-specific speech.
- `Amazon Translate` is a service that you can use to provide translation between multiple languages. You cannot use Amazon Translate to improve transcription for domain-specific speech.
- `Amazon Lex` is an AI service that you can use to create conversational interfaces for applications. Amazon Lex uses natural language understanding and automatic speech recognition to create chatbots. A solution that creates a custom bot in Amazon Lex will not improve transcription accuracy for domain-specific speech.
- `Amazon Q Business` is a generative AI virtual assistant that can answer questions, summarize content, generate content, and complete tasks based on the data that is provided. Amazon Q Business does not provide access to FMs. Amazon Q is not open source.


### Security
-  `Amazon Macie` to discover, classify, and protect sensitive data that is stored in Amazon S3. Macie is useful for data security. However, Macie primarily focuses on data at rest. You cannot use Macie to secure the access and operations of Amazon Bedrock.
- `Amazon Inspector` is a vulnerability management service that continuously scans workloads for software vulnerabilities and unintended network exposure. Amazon Inspector assesses the security and compliance of your AWS resources by performing automated security checks based on best practices and common vulnerabilities. Amazon Inspector can assess EC2 instances and Amazon ECR repositories to provide detailed findings and recommendations for remediation. You can use Amazon Inspector to maintain a secure and compliant AWS environment.
- `AWS Artifact` provides on-demand access to security and compliance documents. AWS Artifact does not identify security vulnerabilities across EC2 instances and Amazon ECR repositories. AWS Artifact does not provide recommendations for remediation.
-  `AWS Config` provides a detailed view of the configuration of AWS resources within your account. AWS Config illustrates the interconnections and historical configurations of your AWS resources. You can use AWS Config to monitor the change of configurations and relationships over time. You cannot use AWS Config to secure the access and operations of Amazon Bedrock.

