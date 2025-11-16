-module(brain_model).
-export([start/0, neuron/2]).

%% To simulate encoding and retrieving information
start() ->
    Neuron1 = spawn(brain_model, neuron, [none, self()]),
    Neuron2 = spawn(brain_model, neuron, [none, Neuron1]),
    Neuron1 ! {connect, Neuron2},

    %% Encoding
    Neuron1 ! {store, perception},

    %% Retrieval
    timer:sleep(1000), %% Simulating a delay for memory consolidation
    Neuron1 ! retrieve,
    receive
        {memory, Memory} ->
            io:format("Retrieved memory: ~p~n", [Memory])
    end.

%% A simplistic model of a neuron that can store and retrieve information
neuron(Memory, ConnectedNeuron) ->
    receive
        {connect, Neuron} ->
            neuron(Memory, Neuron);
        {store, NewMemory} ->
            neuron(NewMemory, ConnectedNeuron);
        retrieve ->
            ConnectedNeuron ! {memory, Memory},
            neuron(Memory, ConnectedNeuron);
        _ ->
            neuron(Memory, ConnectedNeuron)
    end.
