numData = 50;

x = 15*rand(1,numData);
train = (sin(x)+1)/2;

figure(1);
subplot(1,3,1),scatter(x, train);


w1 = 2*rand(30,1)-1;
b1 = 2*rand(30,1)-1;
w2 = 2*rand(1,30)-1;
b2 = 2*rand(1,1)-1;

layer1 = Affine(w1,b1);
layer2 = Sigmoid();
layer3 = Affine(w2,b2);
layer4 = Sigmoid();
layer5 = MSE();

EPOCH = 100000;
LAMBDA = 0.02;
loss = zeros(1,EPOCH);

for epoch = 1:EPOCH
   
    p = layer1.forward(x);
    y = layer2.forward(p);
    p2 = layer3.forward(y);
    y2 = layer4.forward(p2);
    
    loss(epoch) = layer5.forward(y2, train);
    
    dy2 = layer5.backward();
    dp2 = layer4.backward(dy2);
    dy = layer3.backward(dp2);
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);
    
    layer1.update(LAMBDA);
    layer3.update(LAMBDA);
    
end

subplot(1,3,2);
plot(loss)
axis([0 EPOCH 0 max(loss)])
xlabel('Epoch');
ylabel('LOSS');

% Display output graph
xt = [0:0.01:15];
pt = layer1.forward(xt);
yt = layer2.forward(pt);
qt = layer3.forward(yt);
zt = layer4.forward(qt);

subplot(1,3,3);
plot(xt,zt)
