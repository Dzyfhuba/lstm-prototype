import numpy as np
class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for the LSTM cell
        # self.Wf = np.random.randn(hidden_size, hidden_size + input_size)
        # self.bf = np.random.randn(hidden_size, 1)
        self.Wf = np.random.uniform(-np.sqrt(1.0/hidden_size), np.sqrt(1.0/hidden_size), (hidden_size, hidden_size + input_size))
        self.bf = np.zeros((hidden_size, 1))

        # self.Wi = np.random.randn(hidden_size, hidden_size + input_size)
        # self.bi = np.random.randn(hidden_size, 1)
        self.Wi = np.random.uniform(-np.sqrt(1.0/hidden_size), np.sqrt(1.0/hidden_size), (hidden_size, hidden_size + input_size))
        self.bi = np.zeros((hidden_size, 1))

        # self.Wc = np.random.randn(hidden_size, hidden_size + input_size)
        # self.bc = np.random.randn(hidden_size, 1)
        self.Wc = np.random.uniform(-np.sqrt(1.0/hidden_size), np.sqrt(1.0/hidden_size), (hidden_size, hidden_size + input_size))
        self.bc = np.zeros((hidden_size, 1))

        # self.Wo = np.random.randn(hidden_size, hidden_size + input_size)
        # self.bo = np.random.randn(hidden_size, 1)
        self.Wo = np.random.uniform(-np.sqrt(1.0/hidden_size), np.sqrt(1.0/hidden_size), (hidden_size, hidden_size + input_size))
        self.bo = np.zeros((hidden_size, 1))

        # Initialize weights for the output layer
        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.random.randn(output_size, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x, h_prev, C_prev):
      assert h_prev.shape == (self.hidden_size, 1)
      assert x.shape == (self.input_size, 1)
      assert C_prev.shape == (self.hidden_size, 1)

      # Concatenate hidden state and input
      combined = np.concatenate((h_prev, x), axis=0)

      # Forget gate
      ft = self.sigmoid(np.dot(self.Wf, combined) + self.bf)

      # Input gate
      it = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
      Ct_hat = self.tanh(np.dot(self.Wc, combined) + self.bc)

      # Cell state
      Ct = ft * C_prev + it * Ct_hat

      # Output gate
      ot = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
      ht = ot * self.tanh(Ct)

      # Compute output
      yt = np.dot(self.Wy, ht) + self.by

      assert yt.shape == (self.output_size, 1)
      assert ht.shape == (self.hidden_size, 1)
      assert Ct.shape == (self.hidden_size, 1)

      return yt, ht, Ct

    def train(self, X, Y, epochs=1000, learning_rate=0.001):
        m, n_x = X.shape
        n_y, _ = Y.shape

        # Initialize hidden state and cell state
        h_prev = np.zeros((self.hidden_size, 1)) if hasattr(self, 'h_prev') is False else self.h_prev
        C_prev = np.zeros((self.hidden_size, 1)) if hasattr(self, 'C_prev') is False else self.C_prev

        for epoch in range(epochs):
            loss = 0
            for t in range(m):
                x = X[t].reshape(-1, 1)
                y_true = Y[t].reshape(-1, 1)

                # Forward pass
                y_pred, h_prev, C_prev = self.forward(x, h_prev, C_prev)
                self.h_prev = h_prev
                self.C_prev = C_prev

                ## Compute gradients
                dy = y_pred - y_true

                # Compute gradients for output layer
                dWy = np.dot(dy, h_prev.T)
                dby = dy

                # Compute gradients for LSTM cell
                dh = np.dot(self.Wy.T, dy)
                dC = dh * self.tanh(C_prev) * (1 - self.tanh(C_prev))

                # Compute gradients for output gate
                do = dh * self.tanh(C_prev)
                dWo = np.dot(do * self.sigmoid(np.dot(self.Wo, np.concatenate((h_prev, x), axis=0)) + self.bo) * (1 - self.sigmoid(np.dot(self.Wo, np.concatenate((h_prev, x), axis=0)) + self.bo)), np.concatenate((h_prev, x), axis=0).T)
                dbo = do * self.sigmoid(np.dot(self.Wo, np.concatenate((h_prev, x), axis=0)) + self.bo) * (1 - self.sigmoid(np.dot(self.Wo, np.concatenate((h_prev, x), axis=0)) + self.bo))

                # Compute gradients for cell gate
                dC_bar = dC * self.sigmoid(np.dot(self.Wi, np.concatenate((h_prev, x), axis=0)) + self.bi)
                dWc = np.dot(dC_bar * (1 - self.tanh(np.dot(self.Wc, np.concatenate((h_prev, x), axis=0)) + self.bc) ** 2), np.concatenate((h_prev, x), axis=0).T)
                dbc = dC_bar * (1 - self.tanh(np.dot(self.Wc, np.concatenate((h_prev, x), axis=0)) + self.bc) ** 2)

                # Compute gradients for input gate
                di = dC * self.tanh(np.dot(self.Wc, np.concatenate((h_prev, x), axis=0)) + self.bc)
                dWi = np.dot(di * self.sigmoid(np.dot(self.Wi, np.concatenate((h_prev, x), axis=0)) + self.bi) * (1 - self.sigmoid(np.dot(self.Wi, np.concatenate((h_prev, x), axis=0)) + self.bi)), np.concatenate((h_prev, x), axis=0).T)
                dbi = di * self.sigmoid(np.dot(self.Wi, np.concatenate((h_prev, x), axis=0)) + self.bi) * (1 - self.sigmoid(np.dot(self.Wi, np.concatenate((h_prev, x), axis=0)) + self.bi))

                # Compute gradients for forget gate
                df = dC * C_prev
                dWf = np.dot(df * self.sigmoid(np.dot(self.Wf, np.concatenate((h_prev, x), axis=0)) + self.bf) * (1 - self.sigmoid(np.dot(self.Wf, np.concatenate((h_prev, x), axis=0)) + self.bf)), np.concatenate((h_prev, x), axis=0).T)
                dbf = df * self.sigmoid(np.dot(self.Wf, np.concatenate((h_prev, x), axis=0)) + self.bf) * (1 - self.sigmoid(np.dot(self.Wf, np.concatenate((h_prev, x), axis=0)) + self.bf))

                # Update weights and biases
                self.Wy -= learning_rate * dWy
                self.by -= learning_rate * dby
                self.Wo -= learning_rate * dWo
                self.bo -= learning_rate * dbo
                self.Wc -= learning_rate * dWc
                self.bc -= learning_rate * dbc
                self.Wi -= learning_rate * dWi
                self.bi -= learning_rate * dbi
                self.Wf -= learning_rate * dWf
                self.bf -= learning_rate * dbf

                # Compute loss (mean squared error)
                loss += np.sum((y_true - y_pred) ** 2) / 2
            
            printer = f'Epoch {epoch + 1}/{epochs}, Loss: {loss / m}'
            print(printer, end='\r')

    def predict(self, X):
        m, n_x = X.shape
        predictions = np.zeros((m, self.output_size))

        # Initialize hidden state and cell state
        h_prev = np.zeros((self.hidden_size, 1)) if hasattr(self, 'h_prev') is False else self.h_prev
        C_prev = np.zeros((self.hidden_size, 1)) if hasattr(self, 'C_prev') is False else self.C_prev

        for t in range(m):
            x = X[t].reshape(-1, 1)

            # Forward pass
            y_pred, h_prev, C_prev = self.forward(x, h_prev, C_prev)

            # Store prediction
            predictions[t] = y_pred.flatten()

        return predictions

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        mse = np.mean((Y - predictions) ** 2)
        rmse = np.sqrt(mse)
        return mse, rmse