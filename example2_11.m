x = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]';

labels = [1 1 1 1 0 0 0 1];

w = 2*rand(1,3)-1; %random weight
b = 2*rand(1,1)-1; %random bias

layer1 = Affine(w,b);
layer2 = Sigmoid();
layer3 = MSE();

EPOCH = 1000;
LAMBDA = 0.1;

for epoch = 1:EPOCH
    p = layer1.forward(x);
    y = layer2.forward(p);
    loss(epoch) = layer3.forward(y,labels);
    
    dy = layer3.backward();
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);
    
    layer1.update(LAMBDA);
end

figure(1);
plot(loss)
xlabel('Epoch');
ylabel('loss');

layer1.weights

layer1.bias