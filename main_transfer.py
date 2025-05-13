import torch
from transformers import GPT2Tokenizer, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Config

from MedualTime import *
import numpy as np
import json
import pickle
from torch.utils.data import Dataset
import copy
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import math
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
import argparse
from torch.optim.lr_scheduler import StepLR
import random

class InstructionDataset_TimeSeries(Dataset):
    def __init__(self, data_path, partition="train", args = None):
        self.args = args
        with open(data_path + partition+'.pkl', 'rb') as file:
            data = pickle.load(file)
        self.text_ann = data['description']
        self.ts_ann = data['X']
        self.labels = data['Y']
        encoder = LabelEncoder()
        self.labels = encoder.fit_transform(self.labels)
        self.sl_labels = data['sl_Y']
        encoder2 = LabelEncoder()
        self.sl_labels = encoder2.fit_transform(self.sl_labels)
        print('Partition:', partition, ', Size:', len(self.labels))

    def __len__(self):
        return len(self.ts_ann)

    def __getitem__(self, index):
        ts_sample = self.ts_ann[index, ...]
        # index = str(index) # Json requires str for index
        description = self.text_ann[index]
        if not self.args.detection:
            labels = self.labels[index]
        else:
            labels = self.sl_labels[index]

        if not isinstance(description, str):
            description = ' '

        return description, labels, torch.tensor(ts_sample)

class InstructionDataset_TimeSeries_unsupervised(Dataset):
    def __init__(self, data_path,  partition="train",args = None):
        self.args = args
        with open(data_path + partition+'.pkl', 'rb') as file:
            data = pickle.load(file)
        self.text_ann = data['description']
        self.ts_ann = data['X']
        self.labels = data['Y']
        encoder = LabelEncoder()
        self.labels = encoder.fit_transform(self.labels)
        self.sl_labels = data['sl_Y']
        encoder2 = LabelEncoder()
        self.sl_labels = encoder2.fit_transform(self.sl_labels)
        print('Partition:', partition, ', Size:', len(self.labels))

    def __len__(self):
        return len(self.ts_ann)

    def __getitem__(self, index):
        ts_sample = self.ts_ann[index, ...]
        # index = str(index) # Json requires str for index
        description = self.text_ann[index]
        if not self.args.detection:
            labels = self.labels[index]
        else:
            labels = self.sl_labels[index]

        if not isinstance(description, str):
            description = ' '

        return description, self.labels[index], torch.tensor(ts_sample), self.sl_labels[index]




def get_args_parser():
    parser = argparse.ArgumentParser("Dual Multi-modal Modeling", add_help=False)
    parser.add_argument(
        "--add_contrastive_head",
        action="store_true",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gate_fusion",
        action = "store_true",
        help="Use gate fusion to combine the two modalities",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
    )

    parser.add_argument(
        "--test_model_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--patch_len", 
        type=int,
        default=25,
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
    )

    parser.add_argument(
        '--detection',
        action="store_true",
    )
    parser.add_argument(
        "--adapter_layers",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--adapter_len",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--revin",
        action="store_false",
    )

    parser.add_argument("--model_type", type=str, default='', help='Model variant to use')
    parser.add_argument(
        '--gpt_dropout',
        type=float,
        default=0.1,
        help = 'GPT2 dropout rate'
    )
    parser.add_argument(
        "--exp_lr",
        action="store_true",
        help = 'Whether to use exponential learning rate scheduler',
    )
    parser.add_argument('--exp_gamma',        type=float,
        default=0.85,
        help = 'Exponential learning rate scheduler gamma'
    )
    parser.add_argument('--bert_projector', action='store_true', help='Use a linear layer to project the hidden states of the bert model')
    parser.add_argument(
        '--linear_epoch',
        type=int,
        default=500,
        help = 'Linear layer training epoch'
    )
    parser.add_argument('--seed', type=int, default=128, help='Random seed')

    return parser


if __name__ == '__main__':
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args = get_args_parser().parse_args()
    seed = args.seed
    set_seed(seed)
    print(args)
    device = 'cuda:'+str(args.cuda)
    model_name = args.model_type.split('_')[-1]

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.detection:
        log_file_path = f'logs/Transfer_mm_{model_name}_dual_adapter_training_logs_{current_time}.log'
    else:
        log_file_path = f'logs/Transfer_detection_mm_{model_name}_dual_adapter_training_logs{current_time}.log'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file_path,
                        filemode='a')

    args_str = ', '.join(f'{k}={v}' for k, v in vars(args).items())
    logging.info(f'Args Input��{args_str}')
    if args.test_only:
        logging.info(f'Test Only Mode')
        args.num_epochs = 0

    if not args.detection:
        linear_model = torch.nn.Linear(768, 5)
    else:
        linear_model = torch.nn.Linear(768, 4)
    criterion = nn.CrossEntropyLoss()



    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # model = GPT2Model.from_pretrained('gpt2')

    dataset_config = {
    'ts_emb_patch_len': args.patch_len,
     'batch_size': 64,
     'context_points': 1000,
     'patch_len': 25,
     'stride': 25,
     'revin': 1,
    'mask_ratio': 0.0,
     'vars': 12
    }

    dataset_config = SimpleNamespace(**dataset_config)
    config_ = GPT2Config(use_cache=False, adapter_layers = args.adapter_layers, adapter_len = args.adapter_len, adapter_head = 12, attn_pdrop= args.gpt_dropout, embd_pdrop= args.gpt_dropout,
                         resid_pdrop= args.gpt_dropout, bert_projector = args.bert_projector, ts_config = dataset_config, gate_fusion = args.gate_fusion)


    model = MedualTime(config_)

    params_ = torch.load('./gpt2_pretrained/pretrained_gpt2.pth')
    model.load_state_dict(params_, strict=False)
    print('GPT2 params loaded...')

    tuned_list = ['adapter', 'gate']


    for name, param in model.named_parameters():
        if any(tuned in name for tuned in tuned_list):
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(list(linear_model.parameters()) + list(model.parameters()), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3)
    ############################################
    if args.exp_lr:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exp_gamma)


    data_path = './datasets/'
    dataset_train = InstructionDataset_TimeSeries(
        data_path=data_path ,  partition="train", args = args)  
    dataset_val = InstructionDataset_TimeSeries(
        data_path=data_path , partition="val", args = args)  
    dataset_test = InstructionDataset_TimeSeries(
        data_path=data_path , partition="test", args = args)  

    data_loader_train = torch.utils.data.DataLoader( dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    data_loader_test = torch.utils.data.DataLoader( dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    data_loader_val = torch.utils.data.DataLoader( dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = model.to(device)
    linear_model = linear_model.to(device)


    epoch = 0
    best_val_loss = 999999
    step = 0
    num_epochs = args.num_epochs
    val_epoch_list = []
    results_list = []
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        logging.info(f'Epoch {epoch+1}')
        linear_model.train()
        model.train()
        loss_train = 0
        with tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=f'Epoch {epoch + 1}') as tepoch:
        # with enumerate(data_loader_train) as tepoch:
            for i, batch in tepoch:
                # print(i)
                raw_text, labels, ts = batch
                labels = labels.to(device)
                ts = ts.to(device).to(torch.float32)

                # with torch.no_grad():
                output_ts = model.ts_main_forward(raw_text, ts_sample = ts) # keys: last_hidden_state, past_key_values
                hidden_emb_ts = output_ts[:, -1, :] # (batch_size, seq_len, hidden_size)
                output_text = model.text_main_forward(raw_text, ts_sample = ts)
                hidden_emb_text = output_text[:, -1, :] # (batch_size, seq_len, hidden_size)



                hidden_emb = (hidden_emb_ts + hidden_emb_text) / 2
                scores = linear_model(hidden_emb)
                loss = criterion(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                tepoch.set_postfix(loss=loss.item())
                step += 1


        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_train/i:.4f}')
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_train/i:.4f}')
        scheduler.step()

        linear_model.eval()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(enumerate(data_loader_val), total=len(data_loader_val), desc=f'Val Epoch {epoch + 1}', disable=True) as tepoch:
                for i, batch in tepoch:
                # print(i)
                    raw_text, labels, ts = batch
                    labels = labels.to(device)
                    ts = ts.to(device).to(torch.float32)

                    # with torch.no_grad():
                    output_ts = model.ts_main_forward(raw_text, ts_sample=ts)  # keys: last_hidden_state, past_key_values
                    hidden_emb_ts = output_ts[:, -1, :]  # (batch_size, seq_len, hidden_size)
                    output_text = model.text_main_forward(raw_text, ts_sample=ts)
                    hidden_emb_text = output_text[:, -1, :]  # (batch_size, seq_len, hidden_size)


                    hidden_emb = (hidden_emb_ts + hidden_emb_text) / 2

                    scores = linear_model(hidden_emb)
                    val_loss += criterion(scores, labels)
        logging.info(f'Epoch {epoch+1} Val Loss: {val_loss/i:.4f}')
        val_epoch_list.append(val_loss.cpu().numpy())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            linear_model_state_dict = linear_model.state_dict()
            model_state_dict = model.state_dict()
            selected_model_state_dict = {k: v for k, v in model.state_dict().items() if
                                          any(tuned in k for tuned in tuned_list)}
            # print('Saving Best Model...')
            combined_state_dict = {'linear_model': linear_model_state_dict, 'model': selected_model_state_dict}
            torch.save(combined_state_dict, f'save_models/combined_mm_model_best_{current_time}.pth')


        all_preds = []
        all_targets = []
        with torch.no_grad():
            # model.eval()
            with tqdm(enumerate(data_loader_test), total=len(data_loader_test), desc=f'Test Epoch {epoch + 1}', disable=True) as tepoch:
                for i, batch in tepoch:
                # print(i)
                    raw_text, labels, ts = batch
                    labels = labels.to(device)
                    ts = ts.to(device).to(torch.float32)

                    output_ts = model.ts_main_forward(raw_text, ts_sample = ts) # keys: last_hidden_state, past_key_values
                    hidden_emb_ts = output_ts[:, -1, :] # (batch_size, seq_len, hidden_size)
                    output_text = model.text_main_forward(raw_text, ts_sample = ts)
                    hidden_emb_text = output_text[:, -1, :] # (batch_size, seq_len, hidden_size)

                    hidden_emb = (hidden_emb_ts + hidden_emb_text) / 2

                    scores = linear_model(hidden_emb)
                    _, predictions = scores.max(1)
                    all_preds.extend(predictions.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())



        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
        precision_individual, recall_individual, f1_individual, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)

        logging.info(f'Accuracy: {accuracy:.4f}')
        logging.info(f'Macro Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        logging.info(f'Per Class Precision: {precision_individual}')
        logging.info(f'Per Class Recall: {recall_individual}')
        logging.info(f'Per Class F1: {f1_individual}')
        logging.info(f'______________________________________________________')
        logging.info(f'______________________________________________________')

        results_list.append([accuracy, precision, recall, f1, precision_individual, recall_individual, f1_individual])
        # writer.add_scalar('Per Class Precision', precision_individual, epoch)


    if not args.test_only:
        best_model = torch.load(f'save_models/combined_mm_model_best_{current_time}.pth')
    else:
        try:
            best_model = torch.load(args.test_model_path)
        except:
            logging.info(f'No model found in {args.test_model_path}')
            exit()


    best_epoch = np.argmin(val_epoch_list)
    accuracy, precision, recall, f1, precision_individual, recall_individual, f1_individual = results_list[best_epoch]
    logging.info(f'______________________________________________________')
    logging.info(f'______________________________________________________')
    logging.info(f'Best Model Test Results')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Macro Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    logging.info(f'Per Class Precision: {precision_individual}')
    logging.info(f'Per Class Recall: {recall_individual}')
    logging.info(f'Per Class F1: {f1_individual}')
    logging.info(f'______________________________________________________')
    logging.info(f'______________________________________________________')



    print('Embedding...')
    logging.info('Embedding...')
    train_emb = []
    train_labels = []
    train_sl_labels = []
    dataset_train = InstructionDataset_TimeSeries_unsupervised(
        data_path=data_path ,  partition="train", args = args)  
    dataset_val = InstructionDataset_TimeSeries_unsupervised(
        data_path=data_path ,  partition="val", args = args) 
    dataset_test = InstructionDataset_TimeSeries_unsupervised(
        data_path=data_path , partition="test", args = args)  

    data_loader_train = torch.utils.data.DataLoader( dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    data_loader_test = torch.utils.data.DataLoader( dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    data_loader_val = torch.utils.data.DataLoader( dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    with torch.no_grad():
        model.eval()
        linear_model.eval()
        for i, batch in enumerate(data_loader_train):
            raw_text, labels, ts, sl_labels = batch
            sl_labels = sl_labels.to(device)
            labels = labels.to(device)
            ts = ts.to(device).to(torch.float32)

            # with torch.no_grad():
            output_ts = model.ts_main_forward(raw_text, ts_sample=ts)  # keys: last_hidden_state, past_key_values
            hidden_emb_ts = output_ts[:, -1, :]  # (batch_size, seq_len, hidden_size)
            output_text = model.text_main_forward(raw_text, ts_sample=ts)
            hidden_emb_text = output_text[:, -1, :]  # (batch_size, seq_len, hidden_size)

            hidden_emb = (hidden_emb_ts + hidden_emb_text) / 2

            train_emb.append(hidden_emb)
            train_labels.append(labels)
            train_sl_labels.append(sl_labels)

    val_emb = []
    val_labels = []
    val_sl_labels = []
    with torch.no_grad():
        model.eval()
        linear_model.eval()
        for i, batch in enumerate(data_loader_val):
            raw_text, labels, ts, sl_labels = batch
            sl_labels = sl_labels.to(device)
            labels = labels.to(device)
            ts = ts.to(device).to(torch.float32)

            # with torch.no_grad():
            output_ts = model.ts_main_forward(raw_text, ts_sample=ts)  # keys: last_hidden_state, past_key_values
            hidden_emb_ts = output_ts[:, -1, :]  # (batch_size, seq_len, hidden_size)
            output_text = model.text_main_forward(raw_text, ts_sample=ts)
            hidden_emb_text = output_text[:, -1, :]  # (batch_size, seq_len, hidden_size)

            hidden_emb = (hidden_emb_ts + hidden_emb_text) / 2

            val_emb.append(hidden_emb)
            val_labels.append(labels)
            val_sl_labels.append(sl_labels)

    test_emb = []
    test_labels = []
    test_sl_labels = []
    with torch.no_grad():
        model.eval()
        linear_model.eval()
        for i, batch in enumerate(data_loader_test):
            raw_text, labels, ts, sl_labels = batch
            sl_labels = sl_labels.to(device)
            labels = labels.to(device)
            ts = ts.to(device).to(torch.float32)

            # with torch.no_grad():
            output_ts = model.ts_main_forward(raw_text, ts_sample=ts)  # keys: last_hidden_state, past_key_values
            hidden_emb_ts = output_ts[:, -1, :]  # (batch_size, seq_len, hidden_size)
            output_text = model.text_main_forward(raw_text, ts_sample=ts)
            hidden_emb_text = output_text[:, -1, :]  # (batch_size, seq_len, hidden_size)



            hidden_emb = (hidden_emb_ts + hidden_emb_text) / 2
   
            test_emb.append(hidden_emb)
            test_labels.append(labels)
            test_sl_labels.append(sl_labels)

    train_emb = torch.cat(train_emb, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    val_emb = torch.cat(val_emb, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    test_emb = torch.cat(test_emb, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    train_sl_labels = torch.cat(train_sl_labels, dim = 0)
    val_sl_labels = torch.cat(val_sl_labels, dim = 0)
    test_sl_labels = torch.cat(test_sl_labels, dim = 0)

    print('Embedding completed...')




    for args.few_shot in [ 5, 10, 15, 20, 50, 100]:

        linear_model = torch.nn.Linear(768, 5).to(device)


        optimizer_linear = optim.Adam(list(linear_model.parameters()) , lr=0.001)
        scheduler_linear = StepLR(optimizer_linear, step_size=50, gamma=0.9)
        criterion_linear = nn.CrossEntropyLoss()


        train_data_ = []
        train_label_ = []
        train_data_view2 = []
        for i in range(train_labels.max()+1):
            index = torch.where(train_labels == i)[0]
            torch.manual_seed(42+i)
            shuffled_index = torch.randperm(index.size(0))
            train_data_.append(train_emb[index[shuffled_index[:int(args.few_shot)]]] )
            train_label_.append(train_labels[index[shuffled_index[:int(args.few_shot)]]] )
        train_data_= torch.cat(train_data_, dim = 0)
        train_label_ = torch.cat(train_label_, dim = 0)


        best_val_loss = 99999
        emb_loader_train = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(train_data_, train_label_), batch_size=128, shuffle=True, num_workers=0, drop_last=False)
        emb_loader_val = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(val_emb, val_labels), batch_size=128, shuffle=False, num_workers=0, drop_last=False)
        emb_loader_test = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(test_emb, test_labels), batch_size=128, shuffle=False, num_workers=0, drop_last=False)

        for epoch in tqdm(range(args.linear_epoch), desc='Epochs'):
            logging.info(f'Linear layer (5 Class) training, Few Shot {args.few_shot} : Epoch {epoch + 1}')
            linear_model.train()
            loss_train = 0
            with tqdm(enumerate(emb_loader_train), total=len(emb_loader_train), desc=f'(5 Class) , Few Shot {args.few_shot} : Epoch {epoch + 1}', disable=True) as tepoch:
                for i, batch in tepoch:
                    hidden_emb, labels = batch
                    hidden_emb = hidden_emb.to(device)
                    labels = labels.to(device)
                    scores = linear_model(hidden_emb)
                    loss = criterion_linear(scores, labels)
                    optimizer_linear.zero_grad()
                    loss.backward()
                    optimizer_linear.step()
                    loss_train += loss.item()
                    tepoch.set_postfix(loss=loss.item())
                    step += 1

            scheduler_linear.step()

            linear_model.eval()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                with tqdm(enumerate(emb_loader_val), total=len(emb_loader_val), desc=f'(5 Class) , Few Shot {args.few_shot} :  Val Epoch {epoch + 1}', disable=True) as tepoch:
                    for i, batch in tepoch:
                        hidden_emb, labels = batch
                        hidden_emb = hidden_emb.to(device)
                        labels = labels.to(device)
                        scores = linear_model(hidden_emb)
                        loss = criterion_linear(scores, labels)
                        val_loss += criterion_linear(scores, labels)
            logging.info(f'Training Linear Layer (5 Class) , Few Shot {args.few_shot} : Epoch {epoch + 1} Val Loss: {val_loss / i:.4f}')



            all_preds = []
            all_targets = []
            with torch.no_grad():
                with tqdm(enumerate(emb_loader_test), total=len(emb_loader_test), desc=f'(5 Class) , Few Shot {args.few_shot} : Test Epoch {epoch + 1}', disable=True) as tepoch:
                    for i, batch in tepoch:
                        hidden_emb, labels = batch
                        hidden_emb = hidden_emb.to(device)
                        labels = labels.to(device)
                        scores = linear_model(hidden_emb)
                        _, predictions = scores.max(1)
                        all_preds.extend(predictions.cpu().numpy())
                        all_targets.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_targets, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
            precision_individual, recall_individual, f1_individual, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)


            logging.info(f'(5 Class) Few Shot {args.few_shot} Accuracy: {accuracy:.4f}')
            logging.info(f'(5 Class) Few Shot {args.few_shot} Macro Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            logging.info(f'(5 Class) Few Shot {args.few_shot} Per Class Precision: {precision_individual}')
            logging.info(f'(5 Class) Few Shot {args.few_shot} Per Class Recall: {recall_individual}')
            logging.info(f'(5 Class) Few Shot {args.few_shot} Per Class F1: {f1_individual}')
            logging.info(f'______________________________________________________')
            logging.info(f'______________________________________________________')



            if val_loss < best_val_loss:
                best_val_loss = val_loss
                linear_model_state_dict = linear_model.state_dict()
                model_state_dict = model.state_dict()
                selected_model_state_dict = {k: v for k, v in model.state_dict().items() if
                                             any(tuned in k for tuned in tuned_list)}
                # print('Saving Best Model for Linear Training...')
                combined_state_dict = {'linear_model': linear_model_state_dict, 'model': selected_model_state_dict}
                # torch.save(combined_state_dict, f'save_models/combined_lr_full_model_best_{current_time}.pth')
                best_5class_results = [accuracy, precision, recall, f1, precision_individual, recall_individual, f1_individual]

        accuracy, precision, recall, f1, precision_individual, recall_individual, f1_individual = best_5class_results

        logging.info(f'______________________________________________________')
        logging.info(f'______________________________________________________')
        logging.info(f'Best Model Test Results (5 Class) Semi_ratio {args.few_shot}')
        logging.info(f'(5 Class) Few Shot {args.few_shot} Accuracy: {accuracy:.4f}')
        logging.info(f'(5 Class) Few Shot {args.few_shot} Macro Precision: {precision:.4f}, (5 Class) Few Shot {args.few_shot} Recall: {recall:.4f}, (5 Class) Few Shot {args.few_shot} F1: {f1:.4f}')
        logging.info(f'(5 Class) Few Shot {args.few_shot} Per Class Precision: {precision_individual}')
        logging.info(f'(5 Class) Few Shot {args.few_shot} Per Class Recall: {recall_individual}')
        logging.info(f'(5 Class) Few Shot {args.few_shot} Per Class F1: {f1_individual}')
        logging.info(f'______________________________________________________')
        logging.info(f'______________________________________________________')




