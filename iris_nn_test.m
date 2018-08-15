iris_data_test = load('iris_test.csv');
idt_x = iris_data_test(:,1:4)';
idt_x_train = iris_data_test(:, 5:end)';

layer1 = Affine(l1w, l1b);
layer2 = Sigmoid();
layer3 = Affine(l3w, l3b);
layer4 = Sigmoid();
layer5 = Affine(l5w, l5b);
layer6 = Sigmoid();
layer7 = MSE();

p = layer1.forward(idt_x);
y = layer2.forward(p);
p2 = layer3.forward(y);
y2 = layer4.forward(p2);
p3 = layer5.forward(y2);
y3 = layer6.forward(p3);
loss = layer7.forward(y3, idt_x_train);

show_error = [y3; idt_x_train];