function res =  evaluate_mesh(V,T,pred,gt)
[A,E] = CORR_calculate_area(T,V);
correct = pred==gt;
res = sum(correct.*E) / A;
end
