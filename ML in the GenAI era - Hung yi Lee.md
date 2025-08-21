# Machine learning in the age of Generative AI 生成式AI時代下的機器學習(2025)

## 0. Resources

- [生成式AI時代下的機器學習(2025)](https://www.youtube.com/playlist?list=PLJV_el3uVTsNZEFAdQsDeOdzAaHTca2Gi)

## 1. Introduction

1. 生成式人工智慧的常見應用
    1. 生成對話、繪圖、影片、音樂
    2. 機器的思考/推理 (Reasoning) 過程
        1. 先根據問題產生相關的資訊(內心小劇場)
        2. 根據產生出的資訊再提供最後答案
    3. AI Agent
        1. 從經驗學習
        2. 使用不同工具
        3. 應用多種能力
2. 生成式人工智慧的基本原理
    1. 輸入被分解成基本單位 (Token)
    2. 根據輸入 (Prompt) ，經過類神經網路 (Neural Network)，每次只產生一個輸出 Token
        1. 實際是產出 Token 的機率分布
        2. 此過方法稱為 Autoregressive Generation
        3. 模型在遇到 End Token 之前會重複生成程稱
        4. 簡單例子
            1. 輸入"今天天氣很"，輸出"好"
            2. 輸入變成"今天天氣很好"，輸出"，"
            3. 輸入變成"今天天氣很好，"...
            4. 重複直到輸出 End Token
    3. Reasoning
        1. 訓練或提示模型在給出答案前先輸出步驟
        2. 輸出步驟過程稱為 Chain-of-Thought
        3. End Token 會因此被延後輸出
3. Neural Network
    1. 宏觀來看就是把一個 function 拆解成多個串聯的 Layer function
    2. 也稱作深度學習 (Deep Learning) (因為有很多層)
    3. 類似於把複雜的問題拆解成很多比較簡單的問題
    4. 由架構/超參數 (architecture/hyperparameter) 和 參數 (parameter) 組成
        1. 模型中 b 代表 參數數量 billion
        2. 參數數量由架構決定
        3. 參數值由訓練決定
    5. Reasoning 實際上就是一種增加網路深度的方式

4. Model Layer
    1. 今天生成式人工智慧的每一層 Layer 通常還會有更多的小 Layer
    2. 例如有 Self-attention Layer 的 Transformer/Transformer model
    3. 有關Transformer, Mamba 和許多其他變種模型的詳細內容會在第三章講到

5. 通用機器學習模型演變
    1. 2018: 編碼器 (Encoder)
        1. 輸入文字輸出向量，模型例如 ELMo, BERT
        2. 輸出的向量需要經過再不同特化模型來進行不同任務
    2. 2020: GPT-3
        1. 輸入文字可以輸出完整文字
        2. 參數需要經過微調 (Fine-tune) 來適應不同任務，但架構不變
    3. 2023
        1. 輸入文字可以輸出完整文字
        2. 可以根據輸入完成不同任務，架構和參數都不需要改變

6. 賦予 Generative AI 新能力的方法
    1. 固定參數: Prompt Engineering, RAG...
    2. 改變參數
        1. Fine-tune
            1. 加入少量的新訓練資料
            2. 難點在於讓模型保有原本的能力
        2. Model Editing
            1. 找出特定參數直接修改
        3. Model Merging
            1. 結合不同模型的參數