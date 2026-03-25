height = 100;
twist = 100;

module wire_strand (height, twist)
{
    union()
    {
        linear_extrude(height = height, center = true, convexity = 10, twist = 0)
        translate([0, 0, 0])
        circle(r = 1, $fn = 100);

        linear_extrude(height = height, center = true, convexity = 10, twist = twist)
        translate([2, 0, 0])
        circle(r = 1,$fn = 100);

        linear_extrude(height = height, center = true, convexity = 10, twist = twist)
        translate([-2, 0, 0])
        circle(r = 1,$fn = 100);

        linear_extrude(height = height, center = true, convexity = 10, twist = twist)
        translate([-1, 1.72, 0])
        circle(r = 1,$fn = 100);

        linear_extrude(height = height, center = true, convexity = 10, twist = twist)
        translate([1, 1.72, 0])
        circle(r = 1,$fn = 100);

        linear_extrude(height = height, center = true, convexity = 10, twist = twist)
        translate([-1, -1.72, 0])
        circle(r = 1,$fn = 100);

        linear_extrude(height = height, center = true, convexity = 10, twist = twist)
        translate([1, -1.72, 0])
        circle(r = 1,$fn = 100);
    }
}

module wires (height, twist)
{
    union ()
    {
        // Ground wire
        linear_extrude(height = height, center = true, convexity = 10, twist = 0)
        translate([0, 0, 0])
        circle(r = 2, $fn = 100);

        // Twisted wires
        translate([8, 0, 0])
        wire_strand(height, twist);

        translate([-8, 0, 0])
        wire_strand(height, twist);
    }
}

difference()
{
    linear_extrude(height = height+1, center = true, convexity = 10, twist = 0)
        translate([0, 0, 0])
        resize([30,10])circle(r=13, $fn= 100);
    wires (height, twist);
}