images = zeros(30,30,30);
for n = 1:9
   temp = imread(sprintf('BenkerDigits/benleung00%s.png', num2str(n)));
   temp = imresize(temp, [30, 30]);
   images(:,:,n) = mat2gray(rgb2gray(temp));
end
for n = 10:30
   temp = imread(sprintf('BenkerDigits/benleung0%s.png', num2str(n)));
   temp = imresize(temp, [30, 30]);
   images(:,:,n) = mat2gray(rgb2gray(temp));
end

x = zeros(30*30, 30);
for n = 1:30
   temp = images(:,:,n);
   x(:,n) = temp(:);
end

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

output = zeros(10,30);

for col = 1:size(y5,2)
   
    cmax = max(y5(:,col));
    output(:,col) = y5(:,col) >= cmax;
    
end

output_matcher = zeros(1,size(y5,2));
for col = 1:size(y5,2)
    
   output_matcher(1,col) = (sum(output(:,col) == train(:,col)) == 10);
    
end
   