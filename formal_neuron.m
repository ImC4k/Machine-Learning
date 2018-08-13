function y = formal_neuron(x, w, h)
    p = w*x;
    y = p>h;
end