class LSTM_linear_before_after(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.fc2 = nn.Linear(input_size, 32)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(32, hidden_size, bidirectional=False)
        self.fc1 = nn.Linear(hidden_size, output_size)
      

        
        self.h0 = torch.zeros(1, 1, hidden_size, device=device)
        
        self.c0 = torch.zeros(1, 1, hidden_size, device=device)

    def forward(self, x):     
        y = self.fc2(x.view(len(x), -1))
        y_relu = F.sigmoid(y)
        out_lstm, (self.h0, self.c0) = self.lstm(y_relu.view(len(x), 1, -1), (self.h0, self.c0))
        out = self.fc1(out_lstm.view(len(x), -1))
        
        return out[-1][0]
