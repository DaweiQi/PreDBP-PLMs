import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from keras.layers import Input, add, Dense, Activation, BatchNormalization, Flatten, \
    Conv1D
from keras.layers import Dropout, MaxPooling1D
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import KFold
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# ProtT5 feature
with open("../Pre-trained_features/Acrs/Acrs_train_P.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_pre_P = np.array([item for item in feature_dict.values()])
with open("../Pre-trained_features/Acrs/Acrs_train_N.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_pre_N = np.array([item for item in feature_dict.values()])
train_pre = np.row_stack((train_pre_P, train_pre_N))
print("train_pre:", train_pre.shape)
# PsePSSM feature
train_pse_pssm_P = pd.read_csv('../PSSM_features/Acrs/Acrs_train_P_pse_pssm.csv')
train_pse_pssm_P = np.array(train_pse_pssm_P)
train_pse_pssm_P = np.insert(train_pse_pssm_P, 0, values=[1 for _ in range(train_pse_pssm_P.shape[0])], axis=1)
print("train_pse_pssm_P:", train_pse_pssm_P.shape)
train_pse_pssm_N = pd.read_csv('../PSSM_features/Acrs/Acrs_train_N_pse_pssm.csv')
train_pse_pssm_N = np.array(train_pse_pssm_N)
train_pse_pssm_N = np.insert(train_pse_pssm_N, 0, values=[0 for _ in range(train_pse_pssm_N.shape[0])], axis=1)
print("train_pse_pssm_N:", train_pse_pssm_N.shape)
train_pse_pssm = np.row_stack((train_pse_pssm_P, train_pse_pssm_N))
print("train_pse_pssm:", train_pse_pssm.shape)
# splicing
train = np.hstack((train_pse_pssm, train_pre))
y_train, X_train = train[:, 0], train[:, 1:]

# ProtT5 feature
with open("../Pre-trained_features/Acrs/Acrs_independent_test_P.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
test_pre_P = np.array([item for item in feature_dict.values()])
print("test_pre_P:", test_pre_P.shape)
with open("../Pre-trained_features/Acrs/Acrs_independent_test_N.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
test_pre_N = np.array([item for item in feature_dict.values()])
print("test_pre_N:", test_pre_N.shape)
test_pre = np.row_stack((test_pre_P, test_pre_N))
print("test_pre:", test_pre.shape)
# PsePSSM feature
test_pse_pssm_P = pd.read_csv('../PSSM_features/Acrs/Acrs_test_P_pse_pssm.csv')
test_pse_pssm_P = np.array(test_pse_pssm_P)
test_pse_pssm_P = np.insert(test_pse_pssm_P, 0, values=[1 for _ in range(test_pse_pssm_P.shape[0])], axis=1)
print("test_pse_pssm_P:", test_pse_pssm_P.shape)
test_pse_pssm_N = pd.read_csv('../PSSM_features/Acrs/Acrs_test_N_pse_pssm.csv')
test_pse_pssm_N = np.array(test_pse_pssm_N)
test_pse_pssm_N = np.insert(test_pse_pssm_N, 0, values=[0 for _ in range(test_pse_pssm_N.shape[0])], axis=1)
print("test_pse_pssm_N:", test_pse_pssm_N.shape)
test_pse_pssm = np.row_stack((test_pse_pssm_P, test_pse_pssm_N))
print("test_pse_pssm:", test_pse_pssm.shape)
# splicing
test = np.hstack((test_pse_pssm, test_pre))
print("test:", test.shape)

# partition features and labels
y_test, X_test = test[:, 0], test[:, 1:]
# transformed as np.array for CNN model
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)

# standardized dataset
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)  # normalize X to 0-1 range
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# def attention_block(x, filters, kernel_size=3):
#     # main path
#     main = Conv1D(filters, kernel_size, padding='same')(x)
#     main = BatchNormalization()(main)
#     main = Activation('relu')(main)
#     # Attention score
#     attn_score = Conv1D(1, kernel_size, padding='same', activation='sigmoid')(main)
#     main = multiply([main, attn_score])
#     return main


def residual_block(x, filters, kernel_size=3):
    # main path
    main = Conv1D(filters, kernel_size, padding='same')(x)
    main = BatchNormalization()(main)
    main = Activation('relu')(main)
    # residual path
    res = Conv1D(filters, kernel_size, padding='same')(main)
    res = BatchNormalization()(res)
    res = Activation('relu')(res)
    # add
    x = add([x, res])
    return x


# CNN model construction
def CNN(X_train, y_train, X_test, y_test):
    inputShape = (1064, 1)
    input = Input(shape=inputShape)

    x = Conv1D(128, 3, strides=2, name='layer_conv1', padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, name='MaxPool1', padding="same")(x)
    x = Dropout(0.15)(x)

    # # add attention mechanism
    # x = attention_block(x, 128)

    x = Conv1D(64, 3, strides=2, name='layer_conv2', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, name='MaxPool2', padding="same")(x)
    x = Dropout(0.15)(x)

    # residual blocks
    x = residual_block(x, 64)

    x = Conv1D(32, 3, strides=2, name='layer_conv3', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, name='MaxPool3', padding="same")(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', name='fc1')(x)
    x = Dropout(0.15)(x)
    x = Dense(2, activation='softmax', name='fc2')(x)

    model = Model(inputs=input, outputs=x, name='Predict')
    # define SGD optimizer
    momentum = 0.5
    sgd = SGD(lr=0.01, momentum=momentum, decay=0.0, nesterov=False)
    # compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # learning deccay setting
    def step_decay(epoch):  # gradually decrease the learning rate
        initial_lrate = 0.1
        drop = 0.6
        epochs_drop = 3.0
        # math.pow base raised to a power
        lrate = initial_lrate * math.pow(drop,
                                         # math.floor Round numbers down to the nearest integer
                                         math.floor((1 + epoch) / epochs_drop))
        return lrate

    lrate = LearningRateScheduler(step_decay)

    # early stop setting
    early_stop = EarlyStopping(monitor='val_accuracy', patience=40, restore_best_weights=True)

    # summary the callbacks_list
    callbacks_list = [lrate, early_stop]
    model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              epochs=200, callbacks=callbacks_list, batch_size=32, verbose=1)
    return model, model_history


# Implementing 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

# result collection list
ACC_collecton = []
SP_collecton = []
SN_collecton = []
F1_collecton = []
PRE_collecton = []
MCC_collecton = []
AUC_collecton = []
AUPR_collecton = []

for train_index, test_index in kf.split(y_train):
    X_train_CV, X_valid_CV = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
    y_train_CV, y_valid_CV = y_train.iloc[train_index], y_train.iloc[test_index]
    model, model_history = CNN(X_train_CV, y_train_CV, X_valid_CV, y_valid_CV)
    predicted_protability = model.predict(X_valid_CV, batch_size=1)
    predicted_class = np.argmax(predicted_protability, axis=1)
    y_true = y_valid_CV

    TP, FP, FN, TN = confusion_matrix(y_true, predicted_class).ravel()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    ACC_collecton.append(ACC)
    SN_collecton.append(TP / (TP + FN))
    SP_collecton.append(TN / (TN + FP))
    SN = TP / (TP + FN)
    PR = TP / (TP + FP)
    PRE_collecton.append(PR)
    F_SCORE = (2 * SN * PR) / (SN + PR)
    F1_collecton.append(F_SCORE)
    MCC = (TP * TN - FP * FN) / math.pow(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 0.5)
    MCC_collecton.append(MCC)

    # Calculate AUC and AUPR
    y_true = y_true.to_numpy().astype(int)
    AUC = roc_auc_score(y_true, predicted_protability[:, 1])
    AUPR = average_precision_score(y_true, predicted_protability[:, 1])
    AUC_collecton.append(AUC)
    AUPR_collecton.append(AUPR)

print('mean_Acc', np.mean(ACC_collecton))
print('mean_Sen', np.mean(SN_collecton))
print('mean_Spe', np.mean(SP_collecton))
print('mean_F_score', np.mean(F1_collecton))
print('mean_Pre', np.mean(PRE_collecton))
print('mean_Mcc', np.mean(MCC_collecton))
print('mean_AUC', np.mean(AUC_collecton))
print('mean_AUPR', np.mean(AUPR_collecton))

# result collection list
Acc = []
Sen = []
Spe = []
Mcc = []
Pre = []
F_score = []
AUC_test = []
AUPR_test = []
model, model_history = CNN(X_train, y_train, X_test, y_test)
# confusion matrix
predicted_protability = model.predict(X_test, batch_size=1)
predicted_class = np.argmax(predicted_protability, axis=1)
y_true = y_test
TP, FP, FN, TN = confusion_matrix(y_true, predicted_class).ravel()
ACC = (TP + TN) / (TP + TN + FP + FN)
Acc.append(ACC)
SN = TP / (TP + FN)
SP = TN / (TN + FP)
Sen.append(TP / (TP + FN))
Spe.append(TN / (TN + FP))
PR = TP / (TP + FP)
Pre.append(PR)
F_SCORE = (2 * SN * PR) / (SN + PR)
F_score.append(F_SCORE)
MCC = (TP * TN - FP * FN) / math.pow(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 0.5)
Mcc.append(MCC)

# Calculate AUC and AUPR for the test set
AUC = roc_auc_score(y_true, predicted_protability[:, 1])
AUPR = average_precision_score(y_true, predicted_protability[:, 1])
AUC_test.append(AUC)
AUPR_test.append(AUPR)

print('Acc', Acc[0])
print('Sen', Sen[0])
print('Spe', Spe[0])
print('F_score', F_score[0])
print('Pre', Pre[0])
print('Mcc', Mcc[0])
print('AUC', AUC_test[0])
print('AUPR', AUPR_test[0])