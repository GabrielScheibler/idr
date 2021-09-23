import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            dims_albedo,
            dims_shading,
            dims_specular,
            weight_norm=True,
            multires_view=0,
            multires=0,
            lightsources=1
    ):
        super().__init__()

        self.mode = mode
        dim_points = 3
        dim_normals = 3
        dim_view_dirs = 3

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dim_view_dirs = input_ch

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dim_points = input_ch

        if self.mode == 'no_view_dir':
            dim_view_dirs = 0
        elif self.mode == 'no_normal':
            dim_normals = 0

        dims = [d_in + feature_vector_size] + dims + [d_out]
        dims_albedo = [dim_points] + dims_albedo + [3]
        dims_shading = [dim_points + feature_vector_size] + dims_shading + [1]
        dims_specular = [dim_points + dim_normals + dim_view_dirs + feature_vector_size] + dims_specular + [3]


        self.num_layers_albedo = len(dims_albedo)
        self.num_layers_shading = len(dims_shading)
        self.num_layers_specular = len(dims_specular)

        self.lights = torch.nn.Parameter(data=torch.Tensor(lightsources, 4), requires_grad=True)
        #self.lights.data.uniform_(-5, 5)
        # 0 , -4 , 3 , ?
        self.lights.data = torch.Tensor([[0,-4,3,0.1]])
        #self.lights.data.fill_(1)
        self.ambient_light = torch.nn.Parameter(data=torch.Tensor(lightsources, 1), requires_grad=True)
        self.ambient_light.data.uniform_(0, 0.4)


        for l in range(0, self.num_layers_albedo - 1):
            out_dim = dims_albedo[l + 1]
            lin = nn.Linear(dims_albedo[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin_albedo" + str(l), lin)

        """for l in range(0, self.num_layers_shading - 1):
            out_dim = dims_shading[l + 1]
            lin = nn.Linear(dims_shading[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin_shading" + str(l), lin)"""

        for l in range(0, self.num_layers_specular - 1):
            out_dim = dims_specular[l + 1]
            lin = nn.Linear(dims_specular[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin_specular" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors, implicit_network, ray_tracer, sample_network):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        pointpos = points
        pointpos = torch.unsqueeze(pointpos, 0)

        if self.embed_fn is not None:
            points = self.embed_fn(points)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        #x = rendering_input
        x_albedo = torch.cat([points], dim=-1)
        x_shading = torch.cat([points, feature_vectors], dim=-1)
        x_specular = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)

        x = x_albedo
        for l in range(0, self.num_layers_albedo - 1):
            lin = getattr(self, "lin_albedo" + str(l))

            x = lin(x)

            if l < self.num_layers_albedo - 2:
                x = self.relu(x)

        y_albedo = self.tanh(x)

        """x = x_shading
        for l in range(0, self.num_layers_shading - 1):
            lin = getattr(self, "lin_shading" + str(l))

            x = lin(x)

            if l < self.num_layers_shading - 2:
                x = self.relu(x)

        y_shading = self.tanh(x)
        y_shading = torch.cat((y_shading, y_shading, y_shading), 1)"""

        lightpositions = self.lights[:,0:3]
        lightstrengths = self.lights[:,3]
        total_luminance = torch.zeros_like(lightstrengths)
        batch_size = 1
        num_pixels = pointpos.shape[1]

        for l in range(0,self.lights.shape[0]):
            lightpos = lightpositions[l, :]
            lightpos = torch.unsqueeze(torch.unsqueeze(lightpos, 0),0)
            ray_dirs = F.normalize(pointpos - lightpos, dim=2)
            lightpos = torch.squeeze(lightpos, 1)



            #implicit_network.eval()
            #with torch.no_grad():
            points, network_object_mask, dists = ray_tracer(sdf=lambda x: checkpoint(implicit_network, x)[:, 0],
                                                                     cam_loc=lightpos,
                                                                     object_mask=torch.ones_like(points[:,0]).bool(),
                                                                     ray_directions=ray_dirs)
            implicit_network.train()

            points = (lightpos.detach().unsqueeze(1) + dists.detach().reshape(batch_size, num_pixels, 1) * ray_dirs.detach()).reshape(-1, 3)

            sdf_output = implicit_network(points)[:, 0:1]
            #ray_dirs = ray_dirs.reshape(-1, 3)

            if self.training:
                surface_mask = torch.ones_like(points[:,0]).bool()
                surface_points = points[surface_mask]
                surface_dists = dists[surface_mask].unsqueeze(-1)
                surface_ray_dirs = ray_dirs.reshape(-1, 3)[surface_mask]
                surface_lightpos = lightpos.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
                surface_output = sdf_output[surface_mask]
                N = surface_points.shape[0]

                points_all = torch.cat([surface_points], dim=0)

                output = implicit_network(surface_points)
                surface_sdf_values = output[:N, 0:1].detach()

                g = implicit_network.gradient(points_all)
                surface_points_grad = g[:N, 0, :].clone().detach()

                differentiable_surface_points = sample_network(surface_output,
                                                                    surface_sdf_values,
                                                                    surface_points_grad,
                                                                    surface_dists,
                                                                    surface_lightpos,
                                                                    surface_ray_dirs)
                #differentiable_surface_points = surface_points

            else:
                surface_mask = network_object_mask
                differentiable_surface_points = points
                grad_theta = None

            light_dist_diff = torch.norm(torch.squeeze(pointpos,0) - differentiable_surface_points, dim=1)
            #light_dist_diff = torch.abs(light_dists_1 - light_dists_2)
            luminance = 1 - 2 * self.tanh(light_dist_diff * 10)
            luminance = luminance * lightstrengths[l]
            total_luminance = total_luminance + luminance

        total_luminance = total_luminance + torch.squeeze(self.ambient_light)
        y_shading = torch.min(torch.max(total_luminance,torch.ones_like(total_luminance)*-1),torch.ones_like(total_luminance))
        #y_shading = self.tanh(total_luminance)
        y_shading = torch.unsqueeze(y_shading,1)
        y_shading = torch.cat((y_shading, y_shading, y_shading), 1)


        x = x_specular
        for l in range(0, self.num_layers_specular - 1):
            lin = getattr(self, "lin_specular" + str(l))

            x = lin(x)

            if l < self.num_layers_specular - 2:
                x = self.relu(x)

        y_specular = self.tanh(x)

        y = ((((y_albedo + 1) / 2) * ((y_shading + 1) / 2)) + ((y_specular + 1) / 2)) * 2 - 1

        print("lights: ", self.lights)
        print("lights_grad: ", self.lights.grad)

        return y, y_albedo, y_shading, y_specular

class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        print(ray_dirs.shape)
        print(cam_loc.shape)

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: checkpoint(self.implicit_network, x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        print("points: " , points.shape)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)
            #differentiable_surface_points = surface_points

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        rgb_albedo = torch.ones_like(points).float().cuda()
        rgb_shading = torch.ones_like(points).float().cuda()
        rgb_specular = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            render_output = self.get_rbg_value(differentiable_surface_points, view)
            rgb_values[surface_mask] = render_output[0]
            rgb_albedo[surface_mask] = render_output[1]
            rgb_shading[surface_mask] = render_output[2]
            rgb_specular[surface_mask] = render_output[3]

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'rgb_albedo': rgb_albedo,
            'rgb_shading': rgb_shading,
            'rgb_specular': rgb_specular,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals, rgb_albedo, rgb_shading, rgb_specular = self.rendering_network(points, normals, view_dirs, feature_vectors, self.implicit_network, self.ray_tracer, self.sample_network)

        return rgb_vals, rgb_albedo, rgb_shading, rgb_specular
