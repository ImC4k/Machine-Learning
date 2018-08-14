classdef Step < handle
  methods      
    function y = forward(obj, x)
      y = x > 0;
    end
  end
end
