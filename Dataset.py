import csv
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import os


# def plot_bars(x, height):
#     plt.bar(x, height)
#     plt.ylabel('Number of samples')
#     plt.xlabel('Emotions')
#     plt.title('Number of samples per emotion')
#     plt.show()


imUrl = ['G:\\tesina\\Licencias']
Url = 'G:\\tesina\\Licencias\\Results\\CrossVal'
df = pd.read_csv(os.path.join(Url, "CrossVal.csv"))
print(df)
print(df['Estimated Emotion'].unique())
labels = ('Emotion', 'Subject', 'ME_Number')

myFile, myFile2, myFile3 = open('train_data.csv', 'w'), open('val_data.csv', 'w'), open('test_data.csv', 'w')
writer = csv.DictWriter(myFile, fieldnames=labels, lineterminator='\n')
writer2 = csv.DictWriter(myFile2, fieldnames=labels, lineterminator='\n')
writer3 = csv.DictWriter(myFile3, fieldnames=labels, lineterminator='\n')

allEmotions = ['positive', 'negative', 'surprise', 'others']
emotionAddedCount = {'positive': 0, 'negative': 0, 'surprise': 0, 'others': 0}
emotionAddedCount2 = {'positive': 0, 'negative': 0, 'surprise': 0, 'others': 0}
emotionAddedCount3 = {'positive': 0, 'negative': 0, 'surprise': 0, 'others': 0}

counts = df['Estimated Emotion'].value_counts()
emotionCount = {
    'positive': counts['happiness'],
    'negative': counts['disgust'] + counts['fear'] + counts['sadness'],
    'surprise': counts['surprise'],
    'others': counts['others'] + counts['repression']
}
emotionCount2 = {
    'positive': 0,
    'negative': 0,
    'surprise': 0,
    'others': 0
}

emotionCount3 = {
    'positive': 0,
    'negative': 0,
    'surprise': 0,
    'others': 0
}

# plot_bars(allEmotions, emotionCount.values())

print(emotionCount)
for name in allEmotions:
    emotionCount2[name] = round(emotionCount[name] * 0.05)

inicio = 19
for name in allEmotions:
    emotionCount3[name] = emotionCount2[name] * inicio
    if emotionCount3[name] >= emotionCount[name]:
        emotionCount3[name] -= emotionCount[name]

for name in allEmotions:
    emotionCount[name] = round(emotionCount[name] * 0.85)
# df = df.sample(frac=1)
#
# os.makedirs('G:\\tesina\\Licencias\\Results\\CrossVal', exist_ok=True)
#
# df.to_csv('G:\\tesina\\Licencias\\Results\\CrossVal\\CrossVal.csv')

try:
    shutil.rmtree(os.path.join(*imUrl + ['MicroExpressions_Data2']))
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

for _, row in df.iterrows():
    emo = row['Estimated Emotion']
    if emo == 'happiness':
        emo = 'positive'
    elif emo == 'fear' or emo == 'disgust' or emo == 'sadness':
        emo = 'negative'
    elif emo == 'surprise':
        emo = 'surprise'
    else:
        emo = 'others'
    if emotionAddedCount2[emo] < emotionCount2[emo] and emotionAddedCount[emo] + emotionAddedCount3[emo] >= emotionCount3[emo]:
        shutil.copytree(os.path.join(*imUrl + ['Cropped', str(row['Subject']), str(row['Filename'])]),
                        os.path.join(*imUrl + ['MicroExpressions_Data2', 'test', emo, str(row['Subject']),
                                               str(row['Filename'])]))
        writer3.writerow({'Emotion': emo, 'Subject': str(row['Subject']), 'ME_Number': str(row['Filename'])})
        emotionAddedCount2[emo] += 1
    elif emotionAddedCount[emo] < emotionCount[emo]:
        shutil.copytree(os.path.join(*imUrl + ['Cropped', str(row['Subject']), str(row['Filename'])]),
                        os.path.join( *imUrl + ['MicroExpressions_Data2', 'train', emo, str(row['Subject']),
                                                str(row['Filename'])]))
        writer.writerow({'Emotion': emo, 'Subject': str(row['Subject']), 'ME_Number': str(row['Filename'])})
        emotionAddedCount[emo] += 1
    else:
        shutil.copytree(os.path.join(*imUrl + ['Cropped', str(row['Subject']), str(row['Filename'])]),
                        os.path.join(*imUrl + ['MicroExpressions_Data2', 'val', emo, str(row['Subject']),
                                               str(row['Filename'])]))
        writer2.writerow({'Emotion': emo, 'Subject': str(row['Subject']), 'ME_Number': str(row['Filename'])})
        emotionAddedCount3[emo] += 1


# plot_bars(allEmotions, emotionCount.values())
myFile.close()
myFile2.close()