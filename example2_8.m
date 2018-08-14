x = [0,0,1,1;
0,1,0,1];

w = [1.5 1.5;
-1 -0.89];

b = [-0.69;
2];

u = [1 1];

c = [-1];

layer1 = Affine(w,b);
layer2 = Sigmoid();
layer3 = Affine(u,c);
layer4 = Sigmoid();

p = layer1.forward(x);
y = layer2.forward(p)
q = layer3.forward(y);
z = layer4.forward(q)


MSEobj = MSE();
loss = MSEobj.forward(z, [0 1 1 0]);
loss