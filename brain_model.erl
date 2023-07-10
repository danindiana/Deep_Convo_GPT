-module(brain_model).
-export([start_neuron/0, encode/2]).

start_neuron() ->
    spawn(fun neuron/0).

neuron() ->
    receive
        {From, signal} ->
            From ! {self(), received},
            neuron();
        _ ->
            neuron()
    end.

encode(Neuron, SensoryInput) ->
    Neuron ! {self(), signal},
    receive
        {Neuron, received} ->
            io:format("Information encoded~n"),
            ok
    after 5000 -> %% After 5 seconds, if no message is received
        io:format("Failed to encode information~n")
    end.
