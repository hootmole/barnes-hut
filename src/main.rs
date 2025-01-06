use std::{f64::consts::PI, path};
use rand::Rng;
use rayon::prelude::*;
use minifb::{Key, Window, WindowOptions};
use image::{Rgb, RgbImage, ImageBuffer};

type lFloat = f32;
type hFloat = f64;



#[derive(Debug, Clone, Copy)]
struct Point {
    position: (hFloat, hFloat),
    previous_position: (hFloat, hFloat),
    velocity: (hFloat, hFloat),
    mass: lFloat,
}

impl Point {
    fn new(position: (hFloat, hFloat), velocity: (hFloat, hFloat), mass: lFloat) -> Self {
        Point { 
            position, 
            previous_position: (position.0 - velocity.0 * dt, position.1 - velocity.1 * dt),
            velocity,
            mass,
        }
    }
}

fn newtons_gravity(
    point1: (hFloat, hFloat), point2: (hFloat, hFloat),
    mass1: lFloat, mass2: lFloat
) -> (hFloat, hFloat) {
    let dx = (point2.0 - point1.0) * DISTNCE_SCALE;
    let dy = (point2.1 - point1.1) * DISTNCE_SCALE;

    let distance_squared = dx * dx + dy * dy;

    if distance_squared <= CUTOFF * CUTOFF {
        return (0.0, 0.0);
    }

    // Soft distance with epsilon to prevent division by zero
    let soft_distance = (distance_squared + EPSILON * EPSILON).sqrt();
    let force = G * (mass1 * mass2) as hFloat / (soft_distance * soft_distance);

    // Calculate force components
    let fx = force * dx / soft_distance;
    let fy = force * dy / soft_distance;

    (fx, fy)
}

fn distance(p1: (hFloat, hFloat), p2: (hFloat, hFloat)) -> hFloat {
    let dx = (p2.0 - p1.0) * DISTNCE_SCALE;
    let dy = (p2.1 - p1.1) * DISTNCE_SCALE;

    (dx * dx + dy * dy).sqrt()
}

fn add_con(com: (f64, f64, f32), com_to_add: (f64, f64, f32)) -> (f64, f64, f32) {
    let mass_sum = (com.2 + com_to_add.2) as f64;
    
    ((com.0 * com.2 as f64 + com_to_add.0 * com_to_add.2 as f64) / mass_sum, 
    (com.1 * com.2 as f64 + com_to_add.1 * com_to_add.2 as f64) / mass_sum,
    mass_sum as f32
    )
}


#[derive(Debug)]
struct Quadtree {
    boundary: Rectangle,
    point: Option<Point>,
    children: Option<Box<[Quadtree; 4]>>,
    center_of_mass: (hFloat, hFloat),
    total_mass: lFloat,
}

impl Quadtree {
    fn new(boundary: Rectangle) -> Self {
        Quadtree {
            boundary,
            point: None,
            children: None,
            center_of_mass: (0.0, 0.0),
            total_mass: 0.0,
        }
    }

    fn insert(&mut self, point: Point) {
        if !self.boundary.contains(&point) {
            return;
        }
        // update center of mass using weighted average
        self.center_of_mass = (
            (self.center_of_mass.0 * self.total_mass as hFloat + point.position.0 * point.mass as hFloat) / (self.total_mass as hFloat + point.mass as hFloat),
            (self.center_of_mass.1 * self.total_mass as hFloat + point.position.1 * point.mass as hFloat) / (self.total_mass as hFloat + point.mass as hFloat),
        );

        // has point => subdivide and put new point into created children, set the self.point to none
        if let Some(stored_point) = self.point.take() {
            self.subdivide();

            if let Some(ref mut children) = self.children {
                for child in children.iter_mut() {
                    child.insert(stored_point)
                }
            }
        }

        // has no point and no children => insert new point into self.point
        if self.point.is_none() && self.children.is_none() {
            self.point = Some(point);
        } 
        
        else {
            if let Some(ref mut children) = self.children {
                for child in children.iter_mut() {

                    child.insert(point);
                }
            }
        }
    }

    fn subdivide(&mut self) {
        let Rectangle { x, y, width, height } = self.boundary;
        let half_width = width / 2.0;
        let half_height = height / 2.0;

        let nw_boundary = Rectangle::new(x, y, half_width, half_height);
        let ne_boundary = Rectangle::new(x + half_width, y, half_width, half_height);
        let sw_boundary = Rectangle::new(x, y + half_height, half_width, half_height);
        let se_boundary = Rectangle::new(x + half_width, y + half_height, half_width, half_height);

        self.children = Some(Box::new([
            Quadtree::new(nw_boundary),
            Quadtree::new(ne_boundary),
            Quadtree::new(sw_boundary),
            Quadtree::new(se_boundary),
        ]));
    }


    fn compute_force_sum(&self, active_point: &Point) -> (hFloat, hFloat) {
        // this function is intended only to be called on main parent node (the one capturing all other nodes) => it has to have reach to every point
        let mut force_vector = (0.0, 0.0);
        
        // if the main parent has point in it => based on the structure of quadtree that point is the only point => calculate force and return it
        if let Some(ref point) = self.point {
            let force = newtons_gravity(active_point.position, point.position, active_point.mass, point.mass);
            return (force.0 as hFloat, force.1 as hFloat);
        }
        // if the distance between the center of mass and the active point is small enough, approximate the force
        let distance = distance(active_point.position, self.center_of_mass);

        if (self.boundary.width / distance) < DELTA as f64 {
            let force = newtons_gravity(active_point.position, self.center_of_mass, active_point.mass, self.total_mass);
            return (force.0 as hFloat, force.1 as hFloat);
            // println!("aprox done, point count: {}", self.get_pointer_array().len());
            // println!("pos: {:?}, mass: {}", (node_avarage.0, node_avarage.1), node_avarage.2);
        }

        if let Some(ref children) = self.children {
            for child in children.iter() {
                let force = child.compute_force_sum(active_point);
                force_vector.0 += force.0;
                force_vector.1 += force.1;
            }
        }
        force_vector
    }
}

#[derive(Debug, Copy, Clone)]
struct Rectangle {
    x: hFloat,
    y: hFloat,
    width: hFloat,
    height: hFloat,
}

impl Rectangle {
    fn new(x: hFloat, y: hFloat, width: hFloat, height: hFloat) -> Self {
        Rectangle { x, y, width, height }
    }

    fn contains(&self, point: &Point) -> bool {
        point.position.0 >= self.x && point.position.0 <= self.x + self.width &&
        point.position.1 >= self.y && point.position.1 <= self.y + self.height
    }
}

static DELTA: f32 = 0.1; // Body clustering aproximation parameter
static G: f64 = 6.6743e-6;
static CUTOFF: f64 = 0.0001; // Cutoff distance for force calculation
static EPSILON: f64 = 2.0; // Softening parameter
static DISTNCE_SCALE: f64 = 8.0; // Scale factor for distance calculation
static MASS_DIVERSITY: f32 = 0.0; // Mass diversity for random point generation
static dt: f64 = 50.0; // Time step for Verlet integration

fn main() {
    let point_count = 10000;
    let frames = 5000;
    let image_size = (800, 800);

    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    let mut points: Vec<Point> = Vec::new();

    for _i in 1..point_count {
        let radius = rng.gen::<hFloat>() * 45.0;
        let theta: f64 = rng.gen::<hFloat>() * 2.0 * PI;
        let (mut x, mut y) = (theta.cos() * radius, theta.sin() * radius);
        let velocity_scalar = (0.00001 / radius / radius).sqrt();
        let velocity = (-y * velocity_scalar, x * velocity_scalar);
        x += 50.;
        y += 50.;
        let mut point: Point = Point::new((x, y), velocity, 1.0);
        point.mass = rng.gen::<f32>() * MASS_DIVERSITY + 1.0;
        point.velocity = velocity;
        points.push(point);
    }

    // Create the window using minifb
    let mut window = Window::new("Barnes Hut", image_size.0, image_size.1, WindowOptions {
        resize: true,
        scale: minifb::Scale::X1,
        scale_mode: minifb::ScaleMode::AspectRatioStretch,
        ..WindowOptions::default()
    }).expect("Failed to create window");

    window.set_target_fps(60);

    // Create a buffer to store the image data
    let mut buffer: Vec<u32> = vec![0; image_size.0 as usize * image_size.1 as usize];

    for frame in 0..frames {
        let mut quadtree = Quadtree::new(Rectangle::new(0.0, 0.0, 100.0, 100.0));

        for point in points.iter_mut() {
            quadtree.insert(*point);
        }

        buffer.fill(0);

        // Update points and compute forces using Verlet integration
        let updated_points: Vec<Point> = points.par_iter_mut()
        .filter_map(|point| {

            let force = quadtree.compute_force_sum(point);
            let acceleration = (force.0 / point.mass as hFloat, force.1 / point.mass as hFloat);

            point.previous_position = (
                point.position.0 - point.velocity.0 as hFloat * dt,
                point.position.1 - point.velocity.1 as hFloat * dt,
            );

            // Update position using the Verlet integration formula
            let new_position_0 = 2.0 * point.position.0 - point.previous_position.0 + acceleration.0 as hFloat * dt * dt;
            let new_position_1 = 2.0 * point.position.1 - point.previous_position.1 + acceleration.1 as hFloat * dt * dt;

            // Optionally, compute the velocity (for diagnostics or other uses)
            let new_velocity_0 = (new_position_0 - point.previous_position.0) / (2.0 * dt);
            let new_velocity_1 = (new_position_1 - point.previous_position.1) / (2.0 * dt);

            // Update point state
            point.previous_position = point.position; // Save the current position as the previous one
            point.position = (new_position_0, new_position_1); // Update the current position
            point.velocity = (new_velocity_0 as hFloat, new_velocity_1 as hFloat); // Optionally update the velocity (not strictly needed for Verlet)

            // Return the point only if it is within the quadtree boundary
            if quadtree.boundary.contains(point) {
                Some(*point) // Clone or copy the point if needed
            } else {
                None
            }
        })
        .collect();

        // Replace the original points with the filtered and updated ones
        points = updated_points;

        // prepare values for rendering
        let color = [0, 15, 0];
        let packed_color = (color[0] as u32) << 16 | (color[1] as u32) << 8 | color[2] as u32;
        let scale_x = image_size.0 as f64 / 100.0;
        let scale_y = image_size.1 as f64 / 100.0;

        let mut energy_map = vec![0.0; image_size.0 as usize * image_size.1 as usize];

        for point in points.iter() {
            let (pixel_x, pixel_y) = (
                (point.position.0 * scale_x) as u32,
                (point.position.1 * scale_y) as u32,
            );
            // index of the pixel in a 1D array
            let pixel_index = (pixel_y as usize * image_size.0 as usize + pixel_x as usize) as usize;

            let point_velocity_magnitude = (point.velocity.0.powf(2.0) + point.velocity.1.powf(2.0)).sqrt();
            let point_energy = point.mass as f64 * point_velocity_magnitude.powf(2.0) / 2.0;
            energy_map[pixel_index] += point_energy;

            let pixel_color = buffer[pixel_index] + packed_color;

            if true {
                buffer[pixel_index] = 0xFFFFFF;
            } else {
                buffer[pixel_index] = pixel_color
            }
            
        }
        // for (i, energy) in energy_map.iter().enumerate() {
        //     let energy_color = (energy * 5000000.0) as u8;
        //     let r = (energy_color as u32) << 16;
        //     let g = (energy_color as u32) << 8;
        //     let b = energy_color as u32;
        //     buffer[i] = r | g | b;
        // }


        if window.is_key_down(Key::Escape) || !window.is_open() {
            break;
        }
        // Update the window with the new buffer
        window.update_with_buffer(&buffer, image_size.0, image_size.1).unwrap();

        // let img = ImageBuffer::from_fn(image_size.0 as u32, image_size.1 as u32, |x, y| {
        //     let pixel_index = (y as usize * image_size.0 as usize + x as usize) as usize;
        //     let pixel_color = buffer[pixel_index];
        //     let r = ((pixel_color >> 16) & 0xFF) as u8;
        //     let g = ((pixel_color >> 8) & 0xFF) as u8;
        //     let b = (pixel_color & 0xFF) as u8;
        //     Rgb([r, g, b])
        // });
        // let path = path::Path::new("output/").join(format!("frame_{:05}.png", frame));
        // img.save(path).unwrap();

    }
}
