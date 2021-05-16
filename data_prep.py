import os
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import re
path = Path('')
train_labels = pd.read_csv(path / 'train_labels.csv')
tqdm.pandas()
def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')
def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')
def compute_vocab(InChIs):
    special = ['PAD', 'SOS', 'EOS']
    vocab = special + sorted(list({s for InChI in InChIs for s in InChI}))
    return vocab
VOCAB = ['PAD', 'SOS', 'EOS', '(', ')', '+', ',', '-', '/b', '/c', '/h', '/i', '/m', '/s', '/t', '0', '1', '10', 
'100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111', '112', '113', '114', '115', 
'116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', 
'131', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143', '144', '145', '146', 
'147', '148', '149', '15', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '16', '161', '163', 
'165', '167', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', 
'32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', 
'50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', 
'69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', 
'87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 'B', 'Br', 'C', 'Cl', 'D', 'F', 'H', 
'I', 'N', 'O', 'P', 'S', 'Si', 'T']
if __name__ == '__main__':
    train_labels['InChI_1'] = train_labels.InChI.progress_apply(lambda x: x.split('/')[1])
    train_labels['InChI_text'] = train_labels['InChI_1'].progress_apply(split_form) + ' ' + train_labels['InChI'].apply(
                                lambda x: '/'.join(x.split('/')[2:])).progress_apply(split_form2).values
    VOCAB = compute_vocab(train_labels.InChI_text.map(lambda x: x.split(' ')))
    train_labels.to_csv('train_labels_tokenized.csv', index=False)