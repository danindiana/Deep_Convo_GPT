program BrainModel;
type
    Neuron = record
        activation: integer;
    end;
    Layer = array[1..10] of Neuron;
    Network = array[1..10] of Layer;

var
    net: Network;
    i, j, t: integer;

procedure InitializeNet(var net: Network);
begin
    for i := 1 to 10 do
        for j := 1 to 10 do
            net[i, j].activation := 0;
end;

procedure UpdateNeuron(var neuron: Neuron; input: integer);
begin
    if input > 5 then
        neuron.activation := 1
    else
        neuron.activation := 0;
end;

procedure UpdateLayer(var layer: Layer; input: integer);
begin
    for i := 1 to 10 do
        UpdateNeuron(layer[i], input);
end;

procedure UpdateNetwork(var net: Network; input: integer);
begin
    for i := 1 to 10 do
        UpdateLayer(net[i], input);
end;

begin
    InitializeNet(net);
    for t := 1 to 100 do
        UpdateNetwork(net, t mod 10);
end.
