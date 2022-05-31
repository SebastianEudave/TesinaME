import csv
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import os


def plot_bars(x, height):
    plt.bar(x, height)
    plt.ylabel('Number of samples')
    plt.xlabel('Emotions')
    plt.title('Number of samples per emotion')
    plt.show()


imUrl = ['G:\\tesina\\Licencias']
df = pd.read_excel(os.path.join(imUrl[0], "CASME2-coding-20190701.xlsx"))
print(df)
print(df['Estimated Emotion'].unique())
labels = ('Emotion', 'Subject', 'ME_Number')

myFile, myFile2 = open('train_data.csv', 'w'), open('val_data.csv', 'w')
writer = csv.DictWriter(myFile, fieldnames=labels, lineterminator='\n')
writer2 = csv.DictWriter(myFile2, fieldnames=labels, lineterminator='\n')

allEmotions = ['happiness', 'disgust', 'surprise']
emotionAddedCount = {'happiness': 0, 'disgust': 0, 'surprise': 0}

counts = df['Estimated Emotion'].value_counts()
emotionCount = {
    'happiness': counts['happiness'],
    'disgust': counts['disgust'],
    'surprise': counts['surprise']
}

print(emotionCount)
plot_bars(allEmotions, emotionCount.values())

for name in allEmotions:
    emotionCount[name] = round(emotionCount[name] * 0.8)

for _, row in df.iterrows():
    emo = row['Estimated Emotion']
    if emo in allEmotions:
        if emotionAddedCount[emo] < emotionCount[emo]:
            shutil.copytree(os.path.join(*imUrl + ['Cropped', str(row['Subject']), str(row['Filename'])]),
                            os.path.join(*imUrl + ['MicroExpressions_Data2', 'train', emo, str(row['Subject']),
                                                   str(row['Filename'])]))
            writer.writerow({'Emotion': emo, 'Subject': str(row['Subject']), 'ME_Number': str(row['Filename'])})
            emotionAddedCount[emo] += 1
        else:
            shutil.copytree(os.path.join(*imUrl + ['Cropped', str(row['Subject']), str(row['Filename'])]),
                            os.path.join( *imUrl + ['MicroExpressions_Data2', 'val', emo, str(row['Subject']),
                                                    str(row['Filename'])]))
            writer2.writerow({'Emotion': emo, 'Subject': str(row['Subject']), 'ME_Number': str(row['Filename'])})

plot_bars(allEmotions, emotionCount.values())
print(emotionCount)
myFile.close()
myFile2.close()