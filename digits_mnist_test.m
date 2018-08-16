mnist = lost_mnist();
%use MNIST test datasets
test_num = 10000;   % The number of Test samples (MNIST has 10,000 Test samples) 
test_images = mat2gray(mnist.test_images); % normalize images
test_labels = mnist.test_labels;
test_label_vecs = zeros(10,test_num);
for n=1:test_num
  test_label_vecs(test_labels(n)+1, n) = 1; %create test label vectors
end

