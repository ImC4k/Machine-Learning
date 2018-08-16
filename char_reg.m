images = zeros(30,30,30);
for n = 1:9
   temp = imread(sprintf('ChuCheukKiuDigits/cheukkiu00%s.png', num2str(n)));
   temp = imresize(temp, [30, 30]);
   images(:,:,n) = mat2gray(rgb2gray(temp));
end
for n = 10:30
   temp = imread(sprintf('ChuCheukKiuDigits/cheukkiu0%s.png', num2str(n)));
   temp = imresize(temp, [30, 30]);
   images(:,:,n) = mat2gray(rgb2gray(temp));
end

x = zeros(30*30, 30);
for n = 1:30
   temp = images(:,:,n);
   x(:,n) = temp(:);
end

train = [
    1 0 0 0 0 0 0 0 0 0;
    1 0 0 0 0 0 0 0 0 0;
    1 0 0 0 0 0 0 0 0 0;
    0 1 0 0 0 0 0 0 0 0;
    0 1 0 0 0 0 0 0 0 0;
    0 1 0 0 0 0 0 0 0 0;
    0 0 1 0 0 0 0 0 0 0;
    0 0 1 0 0 0 0 0 0 0;
    0 0 1 0 0 0 0 0 0 0;
    0 0 0 1 0 0 0 0 0 0;
    0 0 0 1 0 0 0 0 0 0;
    0 0 0 1 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0 0 0;
    0 0 0 0 1 0 0 0 0 0;
    0 0 0 0 1 0 0 0 0 0;
    0 0 0 0 0 1 0 0 0 0;
    0 0 0 0 0 1 0 0 0 0;
    0 0 0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 0 0 1;
    0 0 0 0 0 0 0 0 0 1;
    0 0 0 0 0 0 0 0 0 1;]';

w1 = 2*rand(700, 900)-1;
b1 = 2*rand(700, 1)-1;
w2 = 2*rand(500, 700)-1;
b2 = 2*rand(500, 1)-1;
w3 = 2*rand(300, 500)-1;
b3 = 2*rand(300, 1)-1;
w4 = 2*rand(100, 300)-1;
b4 = 2*rand(100, 1)-1;
w5 = 2*rand(10, 100)-1;
b5 = 2*rand(10, 1)-1;

layer1 = Affine(w1, b1);
layer2 = Sigmoid();
layer3 = Affine(w2, b2);
layer4 = Sigmoid();
layer5 = Affine(w3, b3);
layer6 = Sigmoid();
layer7 = Affine(w4, b4);
layer8 = Sigmoid();
layer9 = Affine(w5, b5);
layer10 = Sigmoid();
layer11 = MSE();

EPOCH = 1;
LAMBDA = 0.0001;
loss = zeros(1,EPOCH);

for epoch = 1:EPOCH
   
    p = layer1.forward(x);
    y = layer2.forward(p);
    p2 = layer3.forward(y);
    y2 = layer4.forward(p2);
    p3 = layer5.forward(y2);
    y3 = layer6.forward(p3);
    p4 = layer7.forward(y3);
    y4 = layer8.forward(p4);
    p5 = layer9.forward(y4);
    y5 = layer10.forward(p5);
    
    loss(epoch) = layer11.forward(y5, train);
%     Loss = loss(epoch)
    
    dy5 = layer11.backward();
    dp5 = layer10.backward(dy5);
    dy4 = layer9.backward(dp5);
    dp4 = layer8.backward(dy4);
    dy3 = layer7.backward(dp4);
    dp3 = layer6.backward(dy3);
    dy2 = layer5.backward(dp3);
    dp2 = layer4.backward(dy2);
    dy = layer3.backward(dp2);
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);
    
    layer1.update(LAMBDA);
    layer3.update(LAMBDA);
    layer5.update(LAMBDA);
    layer7.update(LAMBDA);
    layer9.update(LAMBDA);
    
end

w1 = layer1.weights;
b1 = layer1.bias;
w2 = layer3.weights;
b2 = layer3.bias;
w3 = layer5.weights;
b3 = layer5.bias;
w4 = layer7.weights;
b4 = layer7.bias;
w5 = layer9.weights;
b5 = layer9.bias;

figure(1);
plot(loss)
xlabel('Epoch');
ylabel('loss');

char_reg_test
sum(output_matcher)