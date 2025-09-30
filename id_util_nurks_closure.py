import numpy as np
import matplotlib.pyplot as plt

def bspline_basis(u, i, p, knots):
    if p == 0:
        if i < 0 or i + 1 >= len(knots):
            return 0.0
        return 1.0 if knots[i] <= u <= knots[i + 1] else 0.0
    
    if i < 0 or i >= len(knots) - 1:
        return 0.0
    
    term1 = 0.0
    if i + p < len(knots):
        den1 = knots[i + p] - knots[i]
        if den1 > 0:
            term1 = ((u - knots[i]) / den1) * bspline_basis(u, i, p - 1, knots)
    
    term2 = 0.0
    if i + p + 1 < len(knots):
        den2 = knots[i + p + 1] - knots[i + 1]
        if den2 > 0:
            term2 = ((knots[i + p + 1] - u) / den2) * bspline_basis(u, i + 1, p - 1, knots)
    
    return term1 + term2

def custom_interoperations_green_curve(points, kappas, is_closed=False):
    points = np.array(points)
    kappas = np.array(kappas)
    degree = 2  # Change to 2
    num_output_points = 1000
   
    if is_closed and len(points) > degree:
        n = len(points)
        extended_points = np.concatenate((points[n-degree:], points, points[0:degree]))
        extended_kappas = np.concatenate((kappas[n-degree:], kappas, kappas[0:degree]))
        len_extended = len(extended_points)
        knots = np.linspace(-degree / float(n), 1 + degree / float(n), len_extended + 1)
   
        u_fine = np.linspace(0, 1, num_output_points, endpoint=False)
   
        smooth_x = np.zeros(num_output_points)
        smooth_y = np.zeros(num_output_points)
   
        for j, u in enumerate(u_fine):
            num_x, num_y, den = 0.0, 0.0, 0.0
            for i in range(len_extended):
                b = bspline_basis(u, i, degree, knots)
                w = extended_kappas[i] * b
                num_x += w * extended_points[i, 0]
                num_y += w * extended_points[i, 1]
                den += w
            if den > 0:
                smooth_x[j] = num_x / den
                smooth_y[j] = num_y / den
   
        smooth_x = np.append(smooth_x, smooth_x[0])
        smooth_y = np.append(smooth_y, smooth_y[0])
   
    return smooth_x, smooth_y

# Test with flower shape
num_petals = 6
num_u = num_petals * 2  # 12
theta = np.linspace(0, 2 * np.pi, num_u, endpoint=False)
radii = 1.0
amplitude = 0.3
r_base = radii + amplitude * np.sin(num_petals * theta)
x_base_coarse = r_base * np.cos(theta)
y_base_coarse = r_base * np.sin(theta)

boundary_points = list(zip(x_base_coarse, y_base_coarse))
boundary_kappas = [1.0] * len(boundary_points)
smooth_x, smooth_y = custom_interoperations_green_curve(boundary_points, boundary_kappas, is_closed=True)

# Check if closed
print('First and last points match:', np.allclose(smooth_x[0], smooth_x[-1]), np.allclose(smooth_y[0], smooth_y[-1]))

# Plot to visualize
plt.figure()
plt.plot(smooth_x, smooth_y, 'b-')
plt.plot(x_base_coarse, y_base_coarse, 'r--')
plt.scatter(x_base_coarse, y_base_coarse, color='red')
plt.axis('equal')
plt.title('Smoothed Closed Curve with degree 2')
plt.show()

# Output some points to check
print('Smooth X first 5:', smooth_x[:5])
print('Smooth Y first 5:', smooth_y[:5])
print('Smooth X last 5:', smooth_x[-5:])
print('Smooth Y last 5:', smooth_y[-5:])
