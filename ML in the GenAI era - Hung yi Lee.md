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

## 2. AI Agent

1. AI Agent 達成目標的步驟
    1. 人類給予目標
    2. AI Agent observe the Environment (Observation)
    3. AI Agent taking actions to affect the Environment
    4. AI Agent observe the Environment again to see it reach the goal or not
    5. repeat 2 to 4 until it reach the goal
2. 方法1: Reinforcement Learning
    1. 在步驟中加入 Reward 即可
    2. 但訓練出的 AI 會侷限於特定任務
    3. 而且設定 Reward 比較困難
3. 方法2: LLM
    1. 上述達成目標的步驟實際上就類似於LLM現有的能力
        1. turn Observations into input Token
        2. turn output Token into Actions
        3. repeat steps until it reach the goal
    2. LLM 不會被限制於特定任務
    3. 可以有較複雜的 Observations
4. 案例1: 即時互動情境
    1. 使用者可以中途增加輸入， AI 會決定是否要更改正在執行的行為
    2. 參考 <https://arxiv.org/abs/2503.04721v1>
5. AI Agent 關鍵能力
    1. 根據經驗/回饋調整行為
        1. LLM 本身就會依據輸入和輸出的累積，更改下一個輸出
        2. 如果對話累積的太長，也可以另外儲存成 Agent's Memory 後使用 RAG
        3. 參考 <https://arxiv.org/abs/2406.08747>
            1. 參考文章指出對於模型輸出，正向的回饋會比負向回饋來的有幫助
        4. 可以使用自身或其他 LLM 來決定/優化 Agent's Memory
            1. 決定哪些資訊要被儲存，要直接儲存還是整理後再儲存
            2. 整理已經儲存記憶 (Reflection)
            3. 參考 
                1. GraphRAG, HippoRAG
                2. MemGPT
                3. Agent Workflow Memory
                4. Agentic Memory for LLM Agents
    2. 使用工具
        1. 這裡工具定義為: 只知道如何使用，不需要理解原理
        2. 也稱為 Function Call
        3. 使用範例
            1. 在 System Prompt 告訴模型
                1. 可以那些工具
                2. 調用工具時，模型要輸出的格式
                3. 工具回覆的格式
                4. 使用範例
            2. 另外設定在模型輸出特定格式時，調用特定工具，再回傳給模型
        4. 常見工具
            1. 搜尋引擎
            2. 可執行程式語言的環境
            3. 其他模型
            4. 使用模型設計工具
        5. 挑戰
            1. 有太多工具可供使用
                1. 把工具說明 Prompt 儲存在 Agent's Memory
                2. 參考 <https://arxiv.org/abs/2502.11271>
            2. LLM 過度相信工具的回覆，導致錯誤的輸出
                1. 最後的輸出仍會受到原本的訓練資料的影響
                    1. 有文章指出，如果工具給出的結果和 LLM 自身的結果差距過大， LLM 會選擇相信自身的結果
                    2. 有文章指出，和人類提通的資訊相比，模型會偏好模型給的資訊，就算是不同模型也是
                    3. 有文章指出，模型會受資訊的 Meat Data 影響，例如偏好發布時間較新的文章
                2. 模型本身就有可能犯錯
            3. 如何抉擇使用工具比較有效率，還是讓 LLM 直接回覆
    3. 做計劃
        1. 挑戰
            1. 2024年左右的 LLM 本身進行複雜規劃的能力還不夠好
            2. 計畫常常趕不上變化，需要根據當下情況變化
        2. 強化計劃能力的方法
            1. 讓模型在執行某些步驟後判斷接下來的成功率，太低就返回上一步重新規劃
                1. 參考 <https://arxiv.org/abs/2407.01476>
            2. 實際執行時某些步驟可能無法回朔
                1. 讓 LLM 扮演 World Model 模擬實際情況
                2. 參考 <https://arxiv.org/abs/2411.06559>
                3. 現今的 Reasoning Models 就是在做類似的事情
                4. 如何避免 Reasoning Models Overthinking 是一個可以研究的方向
                5. 參考 <https://arxiv.org/abs/2502.08235>