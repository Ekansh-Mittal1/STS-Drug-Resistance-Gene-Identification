import pandas as pd

y_matrix_file = "../ISPY-Data/y_df.txt"
df = pd.read_csv(y_matrix_file, header=0, delimiter='\t')
df.fillna(-1, inplace=True)

columns = df.columns.tolist()

for i in range(len(columns)):
    columns[i] = columns[i].replace(" + ", "+")
    columns[i] = columns[i].replace(" ", "")

dict = dict(zip(df.columns.tolist(), columns))

df = df.rename(columns=dict)

identifiers = list(df['PatientIdentifier'])
drugs = []
response = []


def compress(row):
    for c in df.columns[1:]:
        if row[c] != -1.0:
            drugs.append(c)
            response.append(row[c])


df.apply(compress, axis=1)

print(len(identifiers))
print(len(response))

df = pd.DataFrame({'Patient Identifier': identifiers, 'Drug': drugs, 'Response': response})


'''
df = df[(df['Drug'] == 'Paclitaxel+Ganetespib') |
        (df['Drug'] == 'Paclitaxel+MK-2206') |
        (df['Drug'] == 'Paclitaxel+MK-2206+Trastuzumab') |
        (df['Drug'] == 'Paclitaxel+Neratinib') |
        (df['Drug'] == 'Paclitaxel+Pembrolizumab') |
        (df['Drug'] == 'Paclitaxel')]
'''

df.to_csv('../ISPY-Data/comp_y_df.csv', index=False)


