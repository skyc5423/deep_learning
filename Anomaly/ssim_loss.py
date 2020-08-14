
def ssim_loss(input_img, output_img):
    img_size = input_img.shape[1]
    kernel_size = 21
    step_size = 3
    c1 = 0.01
    c2 = 0.03
    total_step_size = int((img_size - kernel_size + 1) / step_size)
    total_ssim = 0.
    input_img = tf.cast(input_img, tf.float32)
    for i in range(BATCH_SIZE):
        for n in range(total_step_size):
            for m in range(total_step_size):
                patch_input = tf.slice(input_img, [0, m * step_size, m * step_size, 0], [1, kernel_size, kernel_size, 1])
                patch_output = tf.slice(output_img, [0, m * step_size, m * step_size, 0], [1, kernel_size, kernel_size, 1])

                mean_input = tf.reduce_mean(tf.squeeze(patch_input))
                var_input = tf.math.square(tf.math.reduce_std(tf.squeeze(patch_input)))

                mean_output = tf.reduce_mean(tf.squeeze(patch_output))
                var_output = tf.math.square(tf.math.reduce_std(tf.squeeze(patch_output)))

                covar = tf.reduce_sum((patch_input - mean_input) * (patch_output - mean_output)) / (img_size * img_size - 1)

                ssim = (2 * mean_input * mean_output + c1) * (2 * covar + c2) / (tf.math.square(mean_input) + tf.math.square(mean_output) + c1) / (var_input + var_output + c2)

                total_ssim += ssim
    return total_ssim

