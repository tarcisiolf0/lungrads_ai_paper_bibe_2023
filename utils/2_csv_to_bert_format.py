import pandas as pd

def csv_single_to_csv_grouped(df, phase):
    # Converter as colunas para strings
    df['token'] = df['token'].astype(str)
    df['iob_label'] = df['iob_label'].astype(str)

    # Agrupar as palavras e os rótulos por número do laudo
    grouped = df.groupby('report').agg({'token': ' '.join, 'iob_label': ' '.join}).reset_index()

    # Renomear as colunas
    grouped.columns = ['report', 'text', 'iob_labels']

    # Salvar o resultado em um novo arquivo CSV
    file_name = 'data\df_'+ phase +'_tokens_labeled_iob_bert_format_full.csv'
    grouped.to_csv(file_name, index=False)

    return

def main():
    df_train = pd.read_csv('data\df_train_llms_tokens_labeled_iob.csv')
    csv_single_to_csv_grouped(df_train, 'train_llms')
    df_test = pd.read_csv('data\df_test_llms_tokens_labeled_iob.csv')
    csv_single_to_csv_grouped(df_test, 'test_llms')

if __name__ == "__main__":
    main()