mnist = load_mnist();
%use MNIST traininig datasets
train_num = 60000; % The number of Training samples (MNIST has 60,000 Training samples)
train_images = mat2gray(mnist.train_images); % normalize images
train_labels = mnist.train_labels;
train_label_vecs = zeros(10,train_num);
for n=1:train_num
  train_label_vecs(train_labels(n)+1, n) = 1; %create training label vectors
end
