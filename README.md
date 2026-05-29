# -Homework-3-Toxic-Comment-Classification

#下載keras(官網給的
!pip install -q keras-nlp --upgrade

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#輸出右邊input的資料集名稱
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import kagglehub
#kagglehub.dataset_download('<owner>/<dataset-slug>')
import numpy as np

#keras的預設，也可以是jax或torch
os.environ["KERAS_BACKEND"] = "tensorflow"  

import keras
import keras_nlp
import wandb
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbMetricsLogger

#登入W&B，然後開始記錄
wandb.login(key="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")
wandb.init(project="jigsaw-deberta-keras", name="deberta-v3-local-keras-run")
#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

#讀取剛剛抓出來的資料集名稱(順便實驗一下有沒有載到)
train_df = pd.read_csv('/kaggle/input/competitions/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
train_df.head()
test_df = pd.read_csv('/kaggle/input/competitions/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_df.head()

#分為六個種類
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#x是訓練集資料，y是六項的答案，把他們分為35:65的測試集:訓練集，固定隨機種子為42，確保每次都長一樣
X_train, X_val, y_train, y_val = train_test_split(
    train_df['comment_text'].values, 
    train_df[LABELS].values, 
    test_size=0.35, 
    random_state=42
)

#右邊input的model的名稱
MODEL_PATH = "/kaggle/input/models/keras/deberta_v3/keras/deberta_v3_base_en/3"

#建立文字預處理器（長度設為192）
preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
    MODEL_PATH,
    sequence_length=192 
)

#設定說模型為多標籤分類核心(每個欄位的機率獨立)+把label分類丟進去
model = keras_nlp.models.DebertaV3Classifier.from_preset(
    MODEL_PATH,
    preprocessor=preprocessor,
    num_classes=len(LABELS),
    activation="sigmoid" 
)

#編譯模型
model.compile( 

#多標籤是非題專用的損失函數
    loss="binary_crossentropy", 

#learning rate設為2乘以10的-5次方，防止pretrain跑掉
#weight decay設為0.01防止模型overfitting
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=0.01),
    metrics=[keras.metrics.AUC(multi_label=True, name="mean_auc")], 
    jit_compile=True 
)


model.fit(
#把東西餵進模型
    x=X_train, 
    y=y_train,

#每個 Epoch 結束自動考一次模擬考
    validation_data=(X_val, y_val),

#每次餵16份資料
    batch_size=16,

#跑一輪
    epochs=1,
    callbacks=[WandbMetricsLogger()] # 自動把進度丟給 W&B
)

#關閉 W&B 紀錄
wandb.finish()

est_df = pd.read_csv("/kaggle/input/competitions/jigsaw-toxic-comment-classification-challenge/test.csv.zip")

predictions = model.predict(test_df['comment_text'].values, batch_size=16, verbose=1)

submission_df = pd.DataFrame()
submission_df['id'] = test_df['id']
for i, label in enumerate(LABELS):
    submission_df[label] = predictions[:, i]

submission_df.to_csv("submission_deberta_keras.csv", index=False)
wandb.finish()
