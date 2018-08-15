images = zeros(25,25,30);
for n = 1:9
   temp = imread(sprintf('ChuCheukKiuDigits/cheukkiu00%s.png', num2str(n)));
   temp = imresize(temp, [25, 25]);
   temp = mat2gray(rgb2gray(temp));
end
for n = 10:30
   temp = imread(sprintf('ChuCheukKiuDigits/cheukkiu0%s.png', num2str(n)));
   temp = imresize(temp, [25, 25]);
   temp = mat2gray(rgb2gray(temp));
end

x = zeros(25*25, 30);
for n = 1:30
   temp = images(:,:,n);
   x(:,n) = temp(:);
end

train = [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9];

w1 = 2*rand(300, 625)-1;
b1 = 2*rand(300, 1)-1;
w2 = 2*rand(100, 300)-1;
b2 = 2*rand(100, 1)-1;
w3 = 2*rand(50, 100)-1;
b3 = 2*rand(50, 1)-1;
w4 = 2*rand(10, 50)-1;
b4 = 2*rand(10, 1)-1;
% w5 = 2*rand(10, 100)-1;
% b5 = 2*rand(10, 1)-1;

layer1 = Affine(w1, b1);
layer2 = Sigmoid();
layer3 = Affine(w2, b2);
layer4 = Sigmoid();
layer5 = Affine(w3, b3);
layer6 = Sigmoid();
layer7 = MSE();

EPOCH = 100000;
LAMBDA = 0.5;

for epoch = 1:EPOCH
    epoch
   
    p = layer1.forward(x);
    y = layer2.forward(p);
    p2 = layer3.forward(y);
    y2 = layer4.forward(p2);
    p3 = layer5.forward(y2);
    y3 = layer6.forward(p3);
    loss = layer7.forward(y3, train)
    
    dy3 = layer7.backward();
    dp3 = layer6.backward(dy3);
    dy2 = layer5.backward(dp3);
    dp2 = layer4.backward(dy2);
    dy = layer3.backward(dp2);
    dp = layer2.backward(dy);
    dx = layer1.backward(dp);
    
    layer1.update(LAMBDA);
    layer3.update(LAMBDA);
    layer5.update(LAMBDA);
    
end

figure(1);
plot(loss)
xlabel('Epoch');
ylabel('loss');