die "usage: perl quad2lin.pl mesh3 > mesh3lin" unless (@ARGV==1);

($fname) = @ARGV;

open IN, "$fname" or die "could not open $fname\n";

$_ = <IN>;
($np,$ne) = split;
$ne1 = 4*$ne;
print "$np $ne1\n";
for $i (1..$np) {
    $_ = <IN>;
    print $_;
}
for $i (1..$ne) {
    $_ = <IN>;
    @_ = split;
    print "$_[0] $_[3] $_[5]\n";
    print "$_[3] $_[4] $_[5]\n";
    print "$_[3] $_[1] $_[4]\n";
    print "$_[5] $_[4] $_[2]\n";
}
