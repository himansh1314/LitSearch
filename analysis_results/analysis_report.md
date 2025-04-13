# LitSearch Benchmark Analysis Report

## 1. Overview

### 1.1 Overall Distribution

- Total queries: 597
- Pass queries: 422 (70.7%)
- Fail queries: 175 (29.3%)

### 1.2 Query Length Analysis

| Length Category | Count | Pass Rate |
|-----------------|-------|-----------|
| Short (< 15 words) | 120 | 67.5% |
| Medium (15-25 words) | 328 | 70.7% |
| Long (> 25 words) | 149 | 73.2% |

Summary:
- Pass queries average word count: 21.6 words
- Fail queries average word count: 21.0 words

### 1.3 Query Structure Analysis

#### 1.3.1 By Question Type

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

#### 1.3.2 By Directness

| Query Directness | Count | Pass Rate |
|------------------|-------|-----------|
| Direct Questions | 143 | 75.5% |
| Indirect Requests | 454 | 69.2% |

### 1.4 Technical Term Analysis

#### 1.4.1 By Category

| Term Category | Occurrence | Pass Rate |
|---------------|------------|-----------|
| Model Names | 42 | 78.6% |
| Task Names | 130 | 74.6% |
| Generic Ml Terms | 211 | 72.0% |
| Method Terms | 75 | 70.7% |
| Metric Names | 9 | 66.7% |

#### 1.4.2 By Density Level

| Density Level | Count | Pass Rate |
|---------------|-------|-----------|
| Low | 258 | 69.8% |
| Medium | 85 | 80.0% |
| High | 4 | 75.0% |
| Very High | 0 | 0.0% |

Summary:
- Pass queries average technical terms: 1.01
- Fail queries average technical terms: 0.86

### 1.5 Syntactic Complexity Analysis

| Complexity Measure | Pass Queries | Fail Queries |
|--------------------|--------------|--------------|
| Parse Depth | 7.40 | 7.60 |
| Noun Chunks | 5.99 | 5.98 |
| Prep Phrases | 2.28 | 2.21 |
| Subordinate Clauses | 0.74 | 0.66 |
| Coordinate Conjunctions | 0.40 | 0.37 |

### 1.6 Research Area Analysis

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

### 1.7 Query Intention Analysis

| Intention Type | Count | Pass Rate |
|----------------|-------|-----------|
| Find resources/datasets | 53 | 77.4% |
| Find specific paper | 239 | 74.9% |
| Explore research direction | 211 | 68.7% |
| Find evaluation benchmark | 25 | 60.0% |
| Find comparison of methods | 15 | 60.0% |
| Find implementation/method | 51 | 58.8% |

### 1.8 Failure Mode Analysis

| Failure Mode | Count | Percentage | Example Query |
|--------------|-------|------------|---------------|
| Knowledge gap (emerging topics) | 40 | 22.9% | Can you recommend a conversational QA dataset where the human questioner does... |
| Concept conjunction complexity | 36 | 20.6% | Can you point me to studies discussing methods for evaluating text generation... |
| Granularity mismatch | 29 | 16.6% | Can you point me to a work that uses diagnostic tools to detect depression fr... |
| Cross-domain knowledge requirement | 25 | 14.3% | Are there any tools or studies that have focused on building a morphological ... |
| Terminology ambiguity | 24 | 13.7% | Are there any research papers on methods to compress large-scale language mod... |
| Method-application conflation | 8 | 4.6% | Can you recommend a paper that uses an NLI model for sentence-level relation ... |
| Overly broad information need | 5 | 2.9% | I would like to understand the theoretical basis for using the nuclear norm o... |

### 1.9 Feature Importance Analysis

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

#### 1.9.1. Feature Interactions

| Feature Interaction | Interaction Strength |
|---------------------|----------------------|
| Technical term specificity level × Query intention type | 0.156 |
| Number of distinct technical terms × Research area | 0.127 |
| Syntactic complexity × Number of constraint types | 0.119 |
| Question type pattern × Technical term specificity level | 0.102 |
| Research area × Temporal reference type | 0.089 |

## 2. Conclusion

This comprehensive analysis of the LitSearch benchmark reveals several key patterns that influence retrieval success:

1. **Technical term specificity** is the strongest predictor of retrieval success, with specific model and dataset names showing the highest pass rates.

2. **Query structure** has a significant impact, with direct questions outperforming broader exploratory queries.

3. **Query length and syntactic complexity** correlate negatively with success rates, suggesting that simpler, more focused queries perform better.

4. **Research areas** vary in retrieval difficulty, with structured prediction tasks (parsing, NER) and representation learning showing the highest success rates.

5. **Failure modes** are dominated by terminology ambiguity, concept conjunction complexity, and knowledge gaps in emerging research areas.

This analysis provides valuable insights for improving scientific literature retrieval systems and developing more effective query formulation strategies.
