import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AudioEventDetector
from dataset import DummyAudioDataset

def train_model():
    dataset = DummyAudioDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = AudioEventDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), 'audio_event_detector.pth')

if __name__ == '__main__':
    train_model()
