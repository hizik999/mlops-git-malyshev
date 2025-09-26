# Импорт необходимых библиотек
# Успешное выполнение этой ячейки кода подтверждает правильную настройку среды разработки

import numpy as np
import pandas as pd
import re
import pymorphy3
import nltk
import matplotlib.pyplot as plt
# %matplotlib inline
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('punkt_tab')
nltk.download('stopwords')

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
nltk.download('wordnet')



df = pd.read_csv('Lesson_1_user_requests.csv')
df.head()



# Нас интересуют только значения "process_texts" - тексты самих обращений, "sphera" - тема, к которой относится письмо, и "categoriya" - категория обращений
df_text = df.copy(deep=True)
df_text = df_text[['process_texts','sphera', 'categoriya']]
df_text.head(10)



# Ещё один удобный способ посмотреть на количество ненулевых и уникальных значений в таблице по каждой переменной (как и другие статистики по численным ппеременным) - функция describe()
df_text.describe(include='all')



#### ВСТАВЬТЕ КОД СЮДА
df_text.dropna(subset=['categoriya', 'process_texts'], inplace=True)
####

print("Всего строк в таблице без пропущенных значений: ", df_text.shape[0])



#### ВСТАВЬТЕ КОД СЮДА
df_text.drop_duplicates(inplace=True)
####

print(f'Таблца без повторяющихся обращений содержит {df_text.shape[0]} строк.')



plt.figure(figsize=(8,5))
sns.countplot(y='sphera', data=df_text)
plt.title('Распределение по сферам')
##plt.show()




plt.figure(figsize=(8,5))
sns.countplot(y='categoriya', data=df_text)
plt.title('Распределение по категориям')
#plt.show()



counts = df_text['categoriya'].value_counts()
new_categories = counts[counts >= 200].index
df_text = df_text[df_text['categoriya'].isin(new_categories)]

df_text['categoriya'].value_counts()



print(f'Таблца теперь содержит {df_text.shape[0]} строк.')



df_text.head()



import re
from nltk.corpus import stopwords
import pymorphy3

ru_stop = set(stopwords.words("russian"))
morph = pymorphy3.MorphAnalyzer()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    # только буквы (кириллица)
    text = re.sub(r"[^А-Яа-яЁё\s]", " ", text)
    # нижний регистр
    text = text.lower()
    # токенизация
    words = text.split()
    # удаление стоп-слов и коротких токенов
    words = [w for w in words if w not in ru_stop and len(w) > 1]
    # лемматизация
    tokens = [morph.parse(w)[0].normal_form for w in words]
    # джойн обратно в строку
    return " ".join(tokens)

df_text["text_clean"] = df_text["process_texts"].apply(preprocess)
df_text[["process_texts", "text_clean"]].head(10)



# Длина сообщений
df_text['len'] = df_text['text_clean'].apply(lambda x: len(x.split()))
sns.histplot(df_text['len'], bins=30)
plt.title('Распределение длин сообщений (в словах)')
#plt.show()


texts = df_text['text_clean'].values #Извлекаем все тексты обращений


from tensorflow.keras.preprocessing.text import Tokenizer

num_words = 60000
mode = "freq"

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(texts)

xAll = tokenizer.texts_to_matrix(texts, mode=mode)

print(xAll.shape)
print(xAll[0, :20])



from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

classes = df_text["categoriya"].values
nClasses = df_text["categoriya"].nunique()

encoder = LabelEncoder()
encoder.fit(classes)
classesEncoded = encoder.transform(classes)

print("Все классы: ", encoder.classes_)
print("Длина вектора: ", classesEncoded.shape)
print("Начало вектора: ", classesEncoded[:20])

# yAll = to_categorical(classesEncoded, nClasses + 1)
yAll = to_categorical(classesEncoded, nClasses)

print("Форма полученной матрицы: ", yAll.shape)
print(
    "Отдельная строка матрицы для класса "
    + encoder.classes_[classesEncoded[0]] + ":",
    yAll[0]
)



from sklearn.model_selection import train_test_split

xTrain, xVal, yTrain, yVal = train_test_split(
    xAll, yAll,
    test_size=0.2,
    random_state=42
)

print("Train X:", xTrain.shape, "Train y:", yTrain.shape)
print("Val   X:", xVal.shape, "Val   y:", yVal.shape)



#Создаём полносвязную сеть
model01 = Sequential()
#Входной полносвязный слой
model01.add(Dense(100, input_dim=num_words,
                  activation="relu"))
#Слой регуляризации Dropout
model01.add(Dropout(0.4))
#Второй полносвязный слой
model01.add(Dense(100, activation='relu'))
#Слой регуляризации Dropout
model01.add(Dropout(0.4))
#Третий полносвязный слой
model01.add(Dense(100, activation='relu'))
#Слой регуляризации Dropout
model01.add(Dropout(0.4))
#Выходной полносвязный слой
model01.add(Dense(nClasses, activation='softmax'))


model01.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Обучаем сеть на выборке
history = model01.fit(xTrain,
                    yTrain,
                    epochs=20,
                    batch_size=128,
                    validation_data=(xVal, yVal))


plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
#plt.show()


plt.figure(figsize=(14,7))
plt.plot(history.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
#plt.show()


model01.summary()


currPred = model01.predict(xTrain[[11]])
#Определяем номер распознанного класса для каждохо блока слов
currOut = np.argmax(currPred, axis=1)


currOut


encoder.inverse_transform(currOut)


label = np.argmax(yTrain[11], axis=0)

label


yTrain[11]

encoder.inverse_transform([label])


df_1 = pd.DataFrame([xTrain[11]]) #берем матричный вид обращения
list_columns = df_1.columns 
df_1[df_1>0].dropna(axis = 1).columns


print('code is done')