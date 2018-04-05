package reconhecimento;

import java.awt.event.KeyEvent;
import java.util.Scanner;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class Captura {

    public static void main(String args[]) throws FrameGrabber.Exception, InterruptedException {

        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converterMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        camera.start();

        CascadeClassifier detectorFace = new CascadeClassifier("src\\recursos\\haarcascade-frontalface-alt.xml");

        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
        Frame frameCapturado = null;

        Mat imagemColorida = new Mat();

        //------------------------------------------
        //Quantas fotos será precis para capturar
        int numeroAmostras = 25;
        int amostra = 1;  //Contador para chegar no 25 e parar
        System.out.println("Digite seu ID");
        //------------------------------------------

        //------------------------------------------
        //Cadastrar o que o usuário colocar 
        Scanner cadastro = new Scanner(System.in);
        int idPessoa = cadastro.nextInt();
        //------------------------------------------

        //------------------------------------------
        //-------------------------------------------------------------------------------------
        // Esse while serve para ficar capturando imagem ao apertar uma tecla e gravar a foto
        //-------------------------------------------------------------------------------------
        while ((frameCapturado = camera.grab()) != null) {

            imagemColorida = converterMat.convert(frameCapturado);

            //----------------------------------------------------------------------------
            //Converter imagens colocridas em cinzas, pois é mais rápido para o computador
            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
            //----------------------------------------------------------------------------

            RectVector facesDetectadas = new RectVector();

            //-------------------------------------------------------------------------------------
            //Detectar as imagens no tamanho minimo de 150x150 e maximo de 500x500
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));
            //-------------------------------------------------------------------------------------

            if (tecla == null) {

                tecla = cFrame.waitKey(5);
            }

            for (int i = 0; i < facesDetectadas.size(); i++) {

                //-------------------------------------------------------------------------------------
                //Detectar a imagem
                //Todas as faces que ele conseguir detectar ele vai jogar nessa variavél
                Rect dadosFaces = facesDetectadas.get(0);
                rectangle(imagemColorida, dadosFaces, new Scalar(0, 0, 255, 0));
                Mat faceCapturada = new Mat(imagemCinza, dadosFaces);
                //Transformar todas as imagens no mesmo gtamanho 
                resize(faceCapturada, faceCapturada, new Size(160, 160));
                //-------------------------------------------------------------------------------------

                if (tecla == null) {

                    tecla = cFrame.waitKey(5);
                }

                //----------------------------------------------------------
                // verificar se a tecla foi acionada para capturar as fotos
                if (tecla != null) {
                    if (tecla.getKeyChar() == 'q') {
                        if (amostra <= numeroAmostras) {

                            //--------------------------------------------------------------------------------
                            //Gravar as fotos capturadas na pasta com o nome e o  formato
                            imwrite("src\\fotos\\pessoa." + idPessoa + "." + amostra + ".jpg", faceCapturada);
                            System.out.println("Foto " + amostra + " capturada \n");
                            amostra++;
                            //--------------------------------------------------------------------------------
                        }
                    }
                    
                    tecla = null;
                }
                //---------------------------------------------------------
            }

            if (tecla == null) {

                tecla = cFrame.waitKey(20);
            }

            if (cFrame.isVisible()) {

                cFrame.showImage(frameCapturado);
            }

            if (amostra > numeroAmostras) {

                //-----------------------------------------
                //Quando tirar 25 fotos vai fechar o sistema
                break;
            }
        }
        //------------------------------------------

        cFrame.dispose();
        camera.stop();
    }
}
