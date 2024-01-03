-module(gru).
-export([new/2, forward/2]).

-record(gru, {
    Wz :: [[number()]],
    Wr :: [[number()]],
    Wh :: [[number()]],
    bz :: [number()],
    br :: [number()],
    bh :: [number()],
    h_prev :: [number()]
}).

%% Helper functions (using external libraries for efficiency)
-define(E, math).  % Alias for the math module
-import(lists, [append/2]).

%% Generate a random matrix with given dimensions
random_matrix(Rows, Cols) ->
    lists:map(fun(_) -> lists:map(fun(_) -> ?E:random() end, lists:seq(1, Cols)) end, lists:seq(1, Rows)).

%% Generate a random vector with given size
random_vector(Size) ->
    lists:map(fun(_) -> ?E:random() end, lists:seq(1, Size)).

%% Sigmoid activation function
sigmoid(X) ->
    1 / (1 + ?E:exp(-X)).

%% Tanh activation function
tanh(X) ->
    ((?E:exp(X) - ?E:exp(-X)) / (?E:exp(X) + ?E:exp(-X))).

%% Element-wise vector multiplication
multiply(A, B) ->
    lists:zipwith(fun(X, Y) -> X * Y end, A, B).

%% Dot product of two vectors
dot(A, B) ->
    lists:foldl(fun(X, Acc) -> Acc + X * B end, 0, A).

%% Concatenate two vectors
concatenate(A, B) ->
    append(A, B).

%% Create a new GRU with given input and hidden sizes
new(InputSize, HiddenSize) ->
    #gru{
        Wz = random_matrix(HiddenSize, InputSize + HiddenSize),
        Wr = random_matrix(HiddenSize, InputSize + HiddenSize),
        Wh = random_matrix(HiddenSize, InputSize + HiddenSize),
        bz = random_vector(HiddenSize),
        br = random_vector(HiddenSize),
        bh = random_vector(HiddenSize),
        h_prev = random_vector(HiddenSize)
    }.

%% Forward pass of the GRU
forward(GRU = #gru{}, Xt) ->
    Z = concatenate(Xt, GRU#gru.h_prev),
    Zt = sigmoid(dot(GRU#gru.Wz, Z) + GRU#gru.bz),
    Rt = sigmoid(dot(GRU#gru.Wr, Z) + GRU#gru.br),
    HtHat = tanh(dot(GRU#gru.Wh, concatenate(Xt, multiply(Rt, GRU#gru.h_prev))) + GRU#gru.bh),
    Ht = multiply(Zt, GRU#gru.h_prev) + multiply((1 - Zt), HtHat),
    GRU#gru{h_prev = Ht, h_t = Ht}.  % Update hidden state
