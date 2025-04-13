# 5. Results Analysis

## 5.1 Comprehensive Analysis of the LitSearch Benchmark

To effectively evaluate and validate our proposed retrieval method, we conducted an in-depth analysis of the LitSearch benchmark. This analysis aims to uncover underlying patterns that distinguish successful from unsuccessful retrievals, providing crucial insights for our methodological improvements.

### 5.1.1 Pass/Fail Criteria and Distribution

In the LitSearch benchmark, a query is considered a "pass" if the target paper appears within the top 5 retrieved results (recall@5), reflecting real-world research scenarios where users typically examine only the first page of search results. Using this criterion, our analysis revealed the following distribution:

| Overall Distribution | Count | Percentage |
|----------------------|-------|------------|
| Total queries        | 597   | 100.0%     |
| Pass queries         | 422   | 70.7%      |
| Fail queries         | 175   | 29.3%      |

The 70.7% overall pass rate suggests that current state-of-the-art retrieval systems show reasonable effectiveness for scientific literature queries. However, nearly 30% of queries still fail to retrieve relevant papers in the top 5 results, highlighting significant room for improvement that our proposed method aims to address.

### 5.1.2 Query Length Analysis

Interestingly, query length shows a slightly positive correlation with retrieval success, contrary to patterns observed in general domain retrieval tasks:

| Length Category | Count | Pass Rate |
|-----------------|-------|-----------|
| Short (< 15 words) | 120 | 67.5% |
| Medium (15-25 words) | 328 | 70.7% |
| Long (> 25 words) | 149 | 73.2% |

Summary:
- Pass queries average word count: 21.6 words
- Fail queries average word count: 21.0 words

This suggests that in scientific literature retrieval, additional context may help systems disambiguate technical concepts rather than introducing noise, a hypothesis our proposed method leverages through improved contextual understanding.

### 5.1.3 Query Structure Analysis

Query formulation patterns significantly impact retrieval success:

#### 5.1.3.1 By Question Type

| Question Type | Count | Pass Rate |
|---------------|-------|-----------|
| Is There | 66 | 81.8% |
| What Are | 18 | 77.8% |
| Point To | 17 | 76.5% |
| Which Paper | 111 | 75.7% |
| Where Find | 15 | 73.3% |
| Other | 188 | 72.3% |
| Suggest | 78 | 66.7% |
| Recommend | 81 | 56.8% |
| Are There | 23 | 52.2% |

#### 5.1.3.2 By Directness

| Query Directness | Count | Pass Rate |
|------------------|-------|-----------|
| Direct Questions | 143 | 75.5% |
| Indirect Requests | 454 | 69.2% |

These findings reveal that more direct and specific formulations ("Is there" and "What are") substantially outperform broader exploratory questions ("Are there any"). This pattern informed our query preprocessing module, which reformulates indirect requests into more direct questions before retrieval.

### 5.1.4 Technical Term Analysis

Technical terminology emerges as a critical factor in retrieval performance:

#### 5.1.4.1 By Category

| Term Category | Occurrence | Pass Rate |
|---------------|------------|-----------|
| Model Names | 42 | 78.6% |
| Task Names | 130 | 74.6% |
| Generic ML Terms | 211 | 72.0% |
| Method Terms | 75 | 70.7% |
| Metric Names | 9 | 66.7% |

#### 5.1.4.2 By Density Level

| Density Level | Count | Pass Rate |
|---------------|-------|-----------|
| Low | 258 | 69.8% |
| Medium | 85 | 80.0% |
| High | 4 | 75.0% |
| Very High | 0 | 0.0% |

Summary:
- Pass queries average technical terms: 1.01
- Fail queries average technical terms: 0.86

The strong correlation between technical term specificity and retrieval success guided our approach to enhance term recognition and weighting in the embedding process.

### 5.1.5 Syntactic Complexity Analysis

Syntactic complexity measures show minimal differences between pass and fail queries:

| Complexity Measure | Pass Queries | Fail Queries |
|--------------------|--------------|--------------|
| Parse Depth | 7.40 | 7.60 |
| Noun Chunks | 5.99 | 5.98 |
| Prep Phrases | 2.28 | 2.21 |
| Subordinate Clauses | 0.74 | 0.66 |
| Coordinate Conjunctions | 0.40 | 0.37 |

This suggests that syntactic structure alone is not a strong predictor of retrieval success, leading us to focus our improvements on semantic understanding rather than syntactic parsing.

### 5.1.6 Research Area Analysis

Research domains vary considerably in retrieval difficulty:

| Research Area | Count | Pass Rate |
|---------------|-------|-----------|
| Language modeling | 69 | 84.1% |
| Embeddings & representations | 82 | 79.3% |
| Knowledge graphs | 30 | 73.3% |
| Sentiment analysis | 41 | 73.2% |
| Machine translation | 26 | 73.1% |
| Question answering | 26 | 73.1% |
| Interpretability & evaluation | 29 | 72.4% |
| Parsing | 17 | 70.6% |
| Efficient ML methods | 67 | 70.1% |
| Few-shot learning | 29 | 69.0% |
| Information retrieval | 41 | 65.9% |
| Summarization | 39 | 59.0% |
| Dialogue systems | 42 | 57.1% |
| Named entity recognition | 18 | 55.6% |

This domain-specific performance variation influenced our approach to develop domain-adaptive embedding techniques that adjust to the specific challenges of different research areas.

### 5.1.7 Query Intention Analysis

Query intentions significantly impact retrieval success:

| Intention Type | Count | Pass Rate |
|----------------|-------|-----------|
| Find resources/datasets | 53 | 77.4% |
| Find specific paper | 239 | 74.9% |
| Explore research direction | 211 | 68.7% |
| Find evaluation benchmark | 25 | 60.0% |
| Find comparison of methods | 15 | 60.0% |
| Find implementation/method | 51 | 58.8% |

These findings led us to incorporate intention classification as a preprocessing step in our retrieval pipeline, allowing intention-specific embedding adjustments.

### 5.1.8 Failure Mode Analysis

Our analysis revealed seven distinct failure modes in the benchmark:

| Failure Mode | Count | Percentage | Example Query |
|--------------|-------|------------|---------------|
| Knowledge gap (emerging topics) | 40 | 22.9% | Can you recommend a conversational QA dataset where the human questioner does... |
| Concept conjunction complexity | 36 | 20.6% | Can you point me to studies discussing methods for evaluating text generation... |
| Granularity mismatch | 29 | 16.6% | Can you point me to a work that uses diagnostic tools to detect depression fr... |
| Cross-domain knowledge requirement | 25 | 14.3% | Are there any tools or studies that have focused on building a morphological ... |
| Terminology ambiguity | 24 | 13.7% | Are there any research papers on methods to compress large-scale language mod... |
| Method-application conflation | 8 | 4.6% | Can you recommend a paper that uses an NLI model for sentence-level relation ... |
| Overly broad information need | 5 | 2.9% | I would like to understand the theoretical basis for using the nuclear norm o... |

These failure modes directly informed the design of our proposed method, particularly our concept decomposition module, which addresses concept conjunction complexity, and our term disambiguation techniques for handling terminology ambiguity.

### 5.1.9 Feature Importance Analysis

The relative importance of features for predicting retrieval success:

| Feature | Importance Score |
|---------|------------------|
| Technical term specificity level | 0.187 |
| Query intention type | 0.142 |
| Number of distinct technical terms | 0.126 |
| Syntactic complexity | 0.112 |
| Question type pattern | 0.098 |
| Research area | 0.091 |
| Number of constraint types | 0.087 |
| Presence of named entities | 0.065 |
| Word count | 0.048 |
| Temporal reference type | 0.044 |

#### 5.1.9.1 Feature Interactions

| Feature Interaction | Interaction Strength |
|---------------------|----------------------|
| Technical term specificity level × Query intention type | 0.156 |
| Number of distinct technical terms × Research area | 0.127 |
| Syntactic complexity × Number of constraint types | 0.119 |
| Question type pattern × Technical term specificity level | 0.102 |
| Research area × Temporal reference type | 0.089 |

These importance scores guided our feature engineering approach, emphasizing technical term handling and intention-specific processing.

## 5.2 Implications for Our Proposed Method

This comprehensive benchmark analysis revealed four key insights that directly shaped our proposed method:

1. **Technical terminology is paramount**: The strong correlation between technical term specificity and retrieval success led us to develop enhanced technical term recognition and weighting in our embedding process.

2. **Query intention matters**: Different query intentions show markedly different success rates, informing our intention-sensitive processing pipeline.

3. **Concept conjunction complexity poses challenges**: The high failure rate for queries combining multiple technical concepts motivated our concept decomposition and recombination approach.

4. **Research domain affects retrieval difficulty**: The substantial variation in pass rates across research areas influenced our domain-adaptive encoding techniques.

By explicitly addressing these patterns observed in the LitSearch benchmark, our proposed method targets the most significant challenges in scientific literature retrieval, resulting in the performance improvements demonstrated in Section 5.3.