import torch


def add_noise(data, noise_type="gaussian", noise_factor=0.2, salt_prob=0.1, pepper_prob=0.1, min_value=0.0, max_value=1.0, num_outliers=5, magnitude=10):
    noisy_data = data.clone()  

    if noise_type == "gaussian":
        noise = torch.randn_like(data) * noise_factor
        noisy_data = noisy_data + noise

    elif noise_type == "uniform":
        noise = (torch.rand_like(data) - 0.5) * 2 * noise_factor
        noisy_data = noisy_data + noise

    elif noise_type == "salt_pepper":
        salt = torch.rand_like(data) < salt_prob
        pepper = torch.rand_like(data) < pepper_prob
        noisy_data[salt] = max_value  
        noisy_data[pepper] = min_value  

    elif noise_type == "outliers":
        outlier_indices = torch.randint(0, data.numel(), (num_outliers,))  
        outlier_values = torch.randint(-magnitude, magnitude, (num_outliers,)).float()  

        noisy_data.view(-1)[outlier_indices] = outlier_values

    noisy_data = torch.clamp(noisy_data, min_value, max_value)

    return noisy_data

