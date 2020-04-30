function A = directed_cycle_matrix(n)
A = spdiags(my_random_matrix(n,2),0:1,n,n);
A(n,1) = my_random_number;

function r = my_random_number()
r = rand * 2 - 1;
function r = my_random_matrix(m,n)
r = rand(m,n) * 2 - 1;