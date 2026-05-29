# -Homework-3-Toxic-Comment-Classification

# 下載keras(官網給的
!pip install -q keras-nlp --upgrade

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# 輸出右邊input的資料集名稱
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')
import numpy as np
