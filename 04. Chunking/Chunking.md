# Chunking Strategy : RAG에서의 AI 성능 향상

## 목차

1. [RAG?](#rag)
2. [Chunking?](#chunking)
3. [RAG에서 Chunking의 중요성](#rag에서-chunking의-중요성)
4. [Chunking Strategy](#chunking-strategy)
5. [Fixed Character Chunking](#fixed-character-chunking)
6. [Recursive Character Chunking](#recursive-character-chunking)
7. [Document-Specific Chunking](#document-specific-chunking)
8. [Semantic Chunking](#semantic-chunking)
9. [Agentic Chunking](#agentic-chunking)
10. [정리](#정리)

## RAG?

![alt text](image/1.RAG.png)

- RAG는 검색 매커니즘을 LLM 모델과 통합하는 접근 방식이다.
- 검색된 문서를 사용하여 AI 기능을 강화하고, 더 정확하고 맥락적으로 풍부한 응답을 생성한다.

## Chunking?

![alt text](image/2.Chunking.png)

- Chunking은 큰 텍스트 조각을 더 작고 관리하기 쉬운 청크로 나누는 것이다.
- 해당 프로세스는 아래와 같은 2가지 중요한 프로세스가 있다.

1. **Data preparation** : 데이터 소스는 청크 문서로 세분화되어 데이터베이스에 저장된다. 청크 내에 임베딩을 생성하는 경우 데이터베이스는 벡터 스토어가 될 수 있다.
2. **Retrieval** : 사용자가 질문을 하면 시스템은 벡터 검색, 전체 텍스트 검색 또는 두 가지 조합을 사용하여 문서 청크를 검색한다. 해당 프로세스는 사용자의 질의와 가장 관련성의 높은 청크를 식별하여 검색한다.

## RAG에서 Chunking의 중요성

1. **청크가 작으면 정확도 향상** : 청킹을 통해 더 작은 텍스트 세그먼트를 색인하고 검색할 수 있어 관련 문서를 찾는 정확도가 높아진다.
2. **청크가 크면 문맥적 생성 향상** : 모델이 크고 나뉜 문서를 걸러내는 대신 구체적이고 관련성 있는 정보를 활용할 수 있으므로 더 일관되고 문맥적으로 정확한 응답을 제공한다.
3. **확장성 및 성능** : 대용량 데이터를 관리하기 쉬운 조각으로 나누어 병렬로 처리함으로써 계산 부하를 줄이고 RAG의 전반적인 성능을 향상시킨다.

- RAG 시스템에서 적절한 Chunking Size를 설정하는 것이 매우 중요하다

## Chunking Strategy

1. **Fixed Character Sizes** : 간단하고, 직관적이며, 텍스트를 고정된 개수의 문자 단위로 나누는 방식
2. **Recursive Character Text Splitting** : 공백이나 구두점과 같은 구분 기호를 사용하여 문맥적으로 더 의미 있는 chunk를 만드는 방식
3. **Document-Specific Splitting** : PDF나 Markdown 파일과 같은 문서 유형에 맞게 chunk 분할 방법을 조정하는 방식
4. **Semantic Splitting** : 임베딩을 사용하여 의미적 내용에 따라 텍스트를 분할하는 방식
5. **Agentic Splitting** : LLM을 활용하여 콘텐츠와 맥락에 따라 최적의 Chunking을 결정하는 방식

## Fixed Character Chunking

## Recursive Character Chunking

## Document-Specific Chunking

## Semantic Chunking

## Agentic Chunking

## 정리
