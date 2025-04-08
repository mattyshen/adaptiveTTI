import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from os.path import join
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import ast
from tqdm import tqdm
import json

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, num_classes):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads),
            num_layers=num_layers
        )
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer_encoder(embedded)
        output = output.mean(dim=1)
        output = self.linear(output)
        return output
    
class TBMDataset(Dataset):
    """TBM dataset."""

    def __init__(self, data_df, concept_df, is_regression=True, is_df=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        if not is_df:
            self.data = pd.read_csv(data_df)
            self.concepts = pd.read_csv(concept_df)
        else:
            self.data = data_df
            self.concepts = pd.DataFrame(concept_df)
            
        self.is_regression = is_regression
        self.is_df = is_df
        self._create_vocab()
        
        
    def _create_vocab(self):
        if not self.is_df:
            categories = [y for y in self.concepts['Response Mapping'].apply(lambda x: [float(cat) for cat in ast.literal_eval(x).values()])]
        else:
            categories = [y for y in self.concepts['Response Mapping'].apply(lambda x: [float(cat) for cat in x.values()])]
        vocab = set()
        for lst in categories:
            sub_set = set(lst)
            vocab = vocab.union(sub_set)
        
        self.vocab = {v:i for i, v in enumerate(vocab)}
        self.inv_vocab = {i:v for i, v in enumerate(vocab)}
    
    def _get_vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        concepts = self.concepts['Concept Name'].to_list()
        columns = self.data.columns
        inv_col_map = {c:i for i, c in enumerate(columns)}
        concepts_col_ids = [inv_col_map[c] for  c in concepts]
        
        X_t = self.data.iloc[[idx], concepts_col_ids]
        indexed_X_t = X_t.applymap(lambda x: self.vocab.get(x, x)).to_numpy().astype(np.int32)
        
        if self.is_regression:
            y_t = self.data.iloc[[idx], inv_col_map['label']].to_numpy().astype(np.float32)
        else:
            y_t = torch.Tensor(self.data.iloc[[idx], inv_col_map['label']].to_numpy()).type(torch.LongTensor)
        
        sample = {'X': indexed_X_t, 'y': y_t}

        return sample

def train(model, dataloader, device, is_regression):
    if is_regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    total_loss = 0
    for sample_batched in dataloader:
        inputs = sample_batched['X'].squeeze(1).to(device) #[batch_size, seq_len]
        labels = sample_batched['y'].to(device) #[batch_size, 1]
        optimizer.zero_grad()
        outputs = model(inputs) #[batch_size, num_classes]
        if is_regression:
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels.squeeze(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
        
def evaluation(model, dataloader, device, is_regression):
    model.eval()
    total_score = 0
    with torch.no_grad():
        for sample_batched in dataloader:
            inputs = sample_batched['X'].squeeze(1).to(device) #[batch_size, seq_len]
            labels = sample_batched['y'] #[batch_size, 1]

            outputs = model(inputs) #[batch_size, num_classes]
            outputs = outputs.detach().cpu().numpy()
            if is_regression:
                score = mean_squared_error(outputs, labels)
            else:
                y_pred = np.argmax(outputs, axis=1)
                score = accuracy_score(y_pred, labels.squeeze(-1))

            total_score += score
    return total_score / len(dataloader)

def predict(model, dataloader, device, is_regression):
    model.eval()
    with torch.no_grad():
        collect = []
        for sample_batched in dataloader:
            inputs = sample_batched['X'].squeeze(1).to(device) #[batch_size, seq_len]

            outputs = model(inputs) #[batch_size, num_classes]
            outputs = outputs.detach().cpu().numpy()
            if is_regression:
                y_pred = outputs.squeeze(-1)
            else:
                y_pred = np.argmax(outputs, axis=1)
            collect.extend(y_pred)
    return np.array(collect)

def predict_proba(model, dataloader, device, is_regression):
    model.eval()
    with torch.no_grad():
        collect = []
        for sample_batched in dataloader:
            inputs = sample_batched['X'].squeeze(1).to(device) #[batch_size, seq_len]

            outputs = model(inputs) #[batch_size, num_classes]
            outputs = outputs.detach().cpu().numpy()
            collect.extend(outputs)
    return np.array(collect)

def predict_and_get_y(model, dataloader, device, is_regression):
    model.eval()
    with torch.no_grad():
        collect = []
        label_collect = []
        for sample_batched in dataloader:
            inputs = sample_batched['X'].squeeze(1).to(device) #[batch_size, seq_len]
            labels = sample_batched['y'] #[batch_size, 1]
            
            outputs = model(inputs) #[batch_size, num_classes]
            outputs = outputs.detach().cpu().numpy()
            if is_regression:
                y_pred = outputs.squeeze(-1)
            else:
                y_pred = np.argmax(outputs, axis=1)
            collect.extend(y_pred)
            label_collect.extend(labels)
    return np.array(collect), np.array(label_collect)

def predict_proba_and_get_y(model, dataloader, device, is_regression):
    model.eval()
    with torch.no_grad():
        collect = []
        label_collect = []
        for sample_batched in dataloader:
            inputs = sample_batched['X'].squeeze(1).to(device) #[batch_size, seq_len]
            labels = sample_batched['y'] #[batch_size, 1]
            
            outputs = model(inputs) #[batch_size, num_classes]
            outputs = outputs.detach().cpu().numpy()
            collect.extend(outputs)
            label_collect.extend(labels)
    return np.array(collect), np.array(label_collect)

def prepare_tbm_data(data_df, concepts_list, is_regression):
    tbm_train_dataset = TBMDataset(data_df, concepts_list, is_regression=is_regression, is_df=True)
    train_dataloader = DataLoader(tbm_train_dataset, batch_size=128, shuffle=True, num_workers=1)
    vocab_size = tbm_train_dataset._get_vocab_size()
    return vocab_size, train_dataloader
    
def fit(num_epochs, model, dataloader, is_regression, device):
    print("start fitting the model...")
    for epoch in tqdm(range(num_epochs)):
        loss = train(model, dataloader, device, is_regression)
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    return model

def fit_and_save(num_epochs, model, dataloader, is_regression, save_path, device):
    print("start fitting the model...")
    for epoch in tqdm(range(num_epochs)):
        loss = train(model, dataloader, device, is_regression)
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    # Save model checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, save_path)
    return model


def main():
    # Define training config
    gpu = 0
    task_name = "agnews"#"agnews"#"cebab"
    teacher_type = "transformer"
    is_regression = False#True
    num_classes = 4#1
    num_epochs = 150#150(agnews), 300(cebab)

    path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define train, test, concept file paths
    train_file = join(path_to_repo, "data", "tbm_gpt4", "{}_{}".format(task_name, teacher_type), "{}_train_df.csv".format(task_name))
    test_file = join(path_to_repo, "data", "tbm_gpt4", "{}_{}".format(task_name, teacher_type), "{}_test_df.csv".format(task_name))
    concept_file = join(path_to_repo, "data", "tbm_gpt4", "{}_{}".format(task_name, teacher_type), "{}_concepts.csv".format(task_name))
                     
    # load train data
    tbm_train_dataset = TBMDataset(train_file, concept_file, is_regression=is_regression)
    train_dataloader = DataLoader(tbm_train_dataset, batch_size=128, shuffle=True, num_workers=1)
    
    vocab_size = tbm_train_dataset._get_vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleTransformer(vocab_size=vocab_size, embedding_dim=52, num_heads=4, num_layers=2, num_classes=num_classes).to(device)

    # Train
    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, device, is_regression)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    
    model_checkpoint_name = join(path_to_repo, "data", "tbm_gpt4", "{}_{}".format(task_name, teacher_type), "{}_simple_transformer_checkpoint.pth".format(task_name))
    # Save model checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, model_checkpoint_name)
    print("Model checkpoint saved to model_checkpoint.pth")
    
    # Evaluation
    # load test data
    tbm_test_dataset = TBMDataset(test_file, concept_file, is_regression=is_regression)
    test_dataloader = DataLoader(tbm_test_dataset, batch_size=128, shuffle=True, num_workers=1)
    
    # Load the model checkpoint
    model.load_state_dict(torch.load(model_checkpoint_name )['model_state_dict'])
    score = evaluation(model, test_dataloader, device, is_regression)
    y_test_teacher = predict_proba(model, test_dataloader, device, is_regression)

    if is_regression:
        print(f'test mse {score:.4f}')
    else:
        print(f'test acc {score:.4f}')
    
    
if __name__ == "__main__":
    main()