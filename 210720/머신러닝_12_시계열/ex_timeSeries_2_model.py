from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

#################################################################
"""
SimpleRNN 클래스 생성시 return_sequences 인수를 True로 하면 출력 순서열 중
마지막 값만 출력하는 것이 아니라 전체 순서열을 3차원 텐서 형태로 출력하므로 
sequence-to-sequence 문제로 풀 수 있다. 다만 입력 순서열과 출력 순서열의 크기는 같아야 한다.
다만 이 경우에는 다음에 오는 Dense 클래스 객체를 
TimeDistributed wrapper를 사용하여 3차원 텐서 입력을 받을 수 있게 확장해 주어야 한다.
"""
from tensorflow.keras.layers import TimeDistributed

def make_model() :
    model2 = Sequential()
    model2.add(SimpleRNN(10,return_sequences=True,input_shape=(3,1))) #return_sequneces=True => sequences-to-sequences(입력결과모두 시퀀스) 모델 ->차원이 늘어남 Dense랑 만나기전에 차원을 줄여야하는데 rnn의 차원을 줄일때는 timedistributed쓴다.
    model2.add(TimeDistributed(Dense (1, activation="linear")))  # 차원을 줄이는것(하나더 늘어난차원을 입력받는것(?) cf.cnn은 flatten쓴것처럼 #글고 여기서 1은 input_shape의 1과 동일. 속성값임 #숫자 예측이기때문에 역시 linear
    model2.compile(loss='mse', optimizer='sgd') #경사하강법
    model2.summary()
    return model2