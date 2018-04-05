package reconhecimento;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class Treinamento {

    public static void main(String args[]) {

        File diretorio = new File("src//fotos");
        FilenameFilter filtroImagem = new FilenameFilter() {

            @Override
            public boolean accept(File dir, String nome) {
                //retornar os arquivos jpg, gif e img
                return nome.endsWith(".jpg") || nome.endsWith(".gif") || nome.endsWith(".img");
            }
        };

        File[] arquivos = diretorio.listFiles(filtroImagem);
        //Armazenar a quantidade de fotos que tenho na pasta
        MatVector fotos = new MatVector(arquivos.length);

        //Armazenar os identificadores das fotos em 32bits
        Mat rotulos = new Mat(arquivos.length, 1, CV_32SC1);

        //Variavél para armazenar corretamente os rotulos
        IntBuffer rotulosBuffer = rotulos.createBuffer();

        //Variavél para contar as imagens
        int contador = 0;

        for (File imagem : arquivos) {
            //Colocar as fotos na variavél foto
            Mat foto = imread(imagem.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);

            //Quebrar os nomes das fotos quando houver ponto e pegar o primeiro numero
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            //System.out.println(classe);

            //Redirecionar as imagens
            resize(foto, foto, new Size(160, 160));

            // pegar a variavél e identificar quanl foto é de qual classe ou seja de qual identificador
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);

            contador++;
        }

        FaceRecognizer eigenFaces = createEigenFaceRecognizer();
        FaceRecognizer fisherFaces = createFisherFaceRecognizer();
        FaceRecognizer lbph = createLBPHFaceRecognizer();

        //Treinar para saber as fotos e gravar em uma pasta
        eigenFaces.train(fotos, rotulos);
        eigenFaces.save("src\\recursos\\classificadorEigenFaces.yml");
        fisherFaces.train(fotos, rotulos);
        fisherFaces.save("src\\recursos\\classificadorFisherfaces.yml");
        lbph.train(fotos, rotulos);
        lbph.save("src\\recursos\\classificadorLBPHfaces.yml");
       
    }
}
