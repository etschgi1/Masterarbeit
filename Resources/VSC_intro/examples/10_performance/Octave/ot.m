% Octave test
function ot(n=4,d=2)
  autoload("setup_c", file_in_loadpath("wrapper.oct"));
  n, d, setup_c(n,d)
end
