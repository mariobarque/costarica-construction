import torch
import torch.nn.init as init
from sklearn.model_selection import train_test_split


def create_multilayer_perceptron_parameters(input_layer_size, hidden_layer_size, output_layer_size):
    wo = torch.FloatTensor(input_layer_size, hidden_layer_size).type(torch.FloatTensor)
    init.uniform_(wo, 0.0, 1.0)
    wo[:, 0] = float('NaN');
    ws = torch.FloatTensor(hidden_layer_size, output_layer_size).type(torch.FloatTensor)
    init.uniform_(ws, 0.0, 1.0)

    return wo, ws


def forward_pass_test(wo, ws, X):
    return forward_pass(torch.Tensor(wo),
                        torch.Tensor(ws),
                        torch.Tensor(X))


def forward_pass(wo, ws, X):
    hidden_output = torch.nn.Sigmoid()(X.mm(wo)/X.shape[1])
    hidden_output[:, 0] = 1
    output = torch.nn.Sigmoid()(hidden_output.mm(ws)/X.shape[1])
    return hidden_output, output


def update_output_weights(ws, hidden_output, T, output, alpha):
    delta_s = (output - T).mul((1 - output).mul(output))
    hidden_output_trans = hidden_output.transpose(0, 1)
    delta_ws = hidden_output_trans.mm(delta_s)
    new_ws = ws - alpha * delta_ws
    return delta_s, new_ws


def update_hidden_weights(X, wo, ws, delta_s, hidden_output, alpha):
    fac1 = (1 - hidden_output).mul(hidden_output)
    fac2 = delta_s.mm(ws.transpose(0, 1))
    delta_o = fac2.mul(fac1)
    weights_delta_o = X.transpose(0, 1).mm(delta_o)

    new_wo = wo - alpha * weights_delta_o
    new_wo[:, 0] = float('NaN')

    return new_wo


def evaluate_error(prediction, T):
  return sum((prediction - T)**2).numpy()[0]


def get_data(df, prediction_variable):
    T = torch.Tensor(df[prediction_variable].values)
    T = T.reshape(len(T), 1)
    X = torch.Tensor(df.drop('cat', 1).values)

    return X, T


def train(df, prediction_variable, epochs = 50, alpha = 0.1):
    X, T = get_data(df, prediction_variable)
    columns_size = X.shape[1]

    input_layer_size, hidden_layer_size, output_layer_size = columns_size, columns_size, 1
    wo, ws = create_multilayer_perceptron_parameters(input_layer_size, hidden_layer_size, output_layer_size)

    errors = []
    for epoch in range(epochs):
        hidden_output, output = forward_pass(wo, ws, X)

        errors.append(evaluate_error(output, T))

        delta_s, new_ws = update_output_weights(ws, hidden_output, T, output, alpha)
        new_wo = update_hidden_weights(X, wo, ws, delta_s, hidden_output, alpha)
        wo = new_wo
        ws = new_ws

    return errors, wo, ws


###########################################################################################
###########################################################################################
#########################Sandbox - For Testing Only########################################
###########################################################################################
###########################################################################################


def construction_data(dataset):
    df = dataset.get_data_for_model('data/construction-data-processed.csv', N=1000)
    train, test = train_test_split(df, test_size=0.2)

    T = torch.Tensor(train['cat'].values)
    T = T.reshape(len(T), 1)
    X = torch.Tensor(train.drop('cat', 1).values)

    return X, T


def create_data():

    #X, T = construction_data()
    X = torch.Tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 0, 1, 0]])
    T = torch.Tensor([[0], [1], [0], [1]])

    return X, T



def main():

    X, T = create_data()
    columns_size = X.shape[1]

    input_layer_size, hidden_layer_size, output_layer_size = columns_size, columns_size, 1
    epochs = 1000
    learning_rate = 1

    wo, ws = create_multilayer_perceptron_parameters(input_layer_size, hidden_layer_size, output_layer_size)
    #(X, T) = createXORData()
    #wo, ws = create_multilayer_perceptron_parameters(3, 3, 1)

    for epoch in range(epochs):
        hidden_output, output = forward_pass(wo, ws, X)

        error = evaluate_error(output, T);
        if epoch % 100 == 0:
            print("Current epoch: ", epoch, " and error: ", error)

        delta_s, new_ws = update_output_weights(ws, hidden_output, T, output, learning_rate)
        new_wo = update_hidden_weights(X, wo, ws, delta_s, hidden_output, learning_rate)
        wo = new_wo
        ws = new_ws

    print("T ")
    print(T.numpy()[:5])

    print("Output ")
    print(output.numpy()[:5])

    print("wo ")
    print(wo.numpy().shape)

    print("ws ")
    print(ws.numpy().shape)



#print(train)

