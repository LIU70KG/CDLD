import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from create_dataset import DAIC_WOZ
from numpy.random import randint
bert_tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased/')  # 注意此处为本地文件夹, 下载bert-base-uncased到MISA-ours下
random.seed(42)


class MSADataset(Dataset):
    def __init__(self, config):

        ## Fetch dataset
        if "daic-woz" in str(config.data_dir).lower():
            dataset = DAIC_WOZ(config)
        else:
            print("Dataset not defined correctly")
            exit()

        self.mode = config.mode
        self.num_segments = 10
        self.duration = 2
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)
        shample_number = dataset.get_shample_number(config.mode)
        # 计算权重（样本数的倒数）
        shample_number = torch.tensor(shample_number, dtype=torch.float32)
        weights = 1.0 / shample_number
        config.weights = weights / weights.sum()  # 正则化权重，使其和为 1
        if "daic-woz" in str(config.data_dir).lower():
            config.visual_size = self.data[0][0][0].shape[1]
            config.acoustic_size = self.data[0][0][1].shape[1]
            config.txt_size = self.data[0][0][2].shape[1]
        else:
            config.visual_size = self.data[0][0][1].shape[1]
            config.acoustic_size = self.data[0][0][2].shape[1]
        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb


    def _get_fragment(self, record):

        num_frames = record[0][0].shape[0]
        if self.num_segments * self.duration > num_frames:
            offsets = list(range(num_frames))  # 返回0到X-1的所有数字
        # 从0到X-1中随机选择K个不同的数字
        else:
            offsets = random.sample(range(num_frames), self.num_segments * self.duration)
            offsets = sorted(offsets)
        # if record[2]==425:
        #     print(offsets)
        # 挑选帧
        ((visual_fea, audio_fea, txt_fea), label, number) = record
        visual = visual_fea[offsets, :]
        audio = audio_fea[offsets, :]
        txt = txt_fea[offsets, :]
        paragraph_inforamtion = ((visual, audio, txt), label, number)

        return paragraph_inforamtion


    def __getitem__(self, index):
        record = self.data[index]

        if self.mode == 'train':
            segment = self._get_fragment(record)
            return segment
        elif self.mode == 'test':
            segment = self._get_fragment(record)
            return segment
            # return record

        # return record


    def __len__(self):
        return self.len


def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)

    print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)

        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things

        if  batch[0][0].__len__() == 4:  # cmdc与mosi等数据集
            labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
            label_area = torch.floor_divide(labels, 5)
            label_area[labels >= 25] = 4  # 处理标签在25及以上的情况
            label_shifting = labels-(label_area*5+2)

            try:
                if batch[0][0][0].shape[1] == 768 and batch[0][0][1].shape[1] == 768 and batch[0][0][2].shape[1] == 128:  # 其实限定了是CMDC
                    sentences = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch])
            except:  # mosi等数据集
                sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)

            visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
            acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

            ## BERT-based features input prep
            SENT_LEN = sentences.size(0)
            # Create bert indices using tokenizer

            bert_details = []
            for sample in batch:
                if batch[0][0].__len__() == 4:
                    text = " ".join(sample[0][3])
                if batch[0][0].__len__() == 5:
                    text = " ".join(sample[0][4])
                # 原版警告
                # encoded_bert_sent = bert_tokenizer.encode_plus(
                #     text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)

                encoded_bert_sent = bert_tokenizer.encode_plus(
                    text, max_length=SENT_LEN + 2, add_special_tokens=True, truncation=True, padding='max_length')
                bert_details.append(encoded_bert_sent)

            # Bert things are batch_first
            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

            # lengths are useful later in using RNNs
            lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

            return sentences, visual, acoustic, labels, label_area, label_shifting, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask

        if batch[0][0].__len__() == 3:  # DAIC
            labels = [sample[1][1] for sample in batch]
            labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
            label_area = torch.floor_divide(labels, 5)
            label_area[labels >= 25] = 4  # 处理标签在25及以上的情况
            label_shifting = labels-(label_area*5+2)

            visual = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch])
            acoustic = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
            sentences = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

            # lengths are useful later in using RNNs
            lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
            bert_sentences = None
            bert_sentence_types = None
            bert_sentence_att_mask = None
            return sentences, visual, acoustic, labels, label_area, label_shifting, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
